from __future__ import absolute_import

import argparse
import json
import multiprocessing
from asyncio.log import logger
from collections import defaultdict
from os import path

import boto3
import torch
from botocore.exceptions import ClientError
from catalyst.utils import load_checkpoint
from pandas import DataFrame
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from retinopathy.augmentations import get_test_transform
from retinopathy.dataset import get_class_names, RetinopathyDataset
from retinopathy.factory import get_model
from retinopathy.inference import ApplySoftmaxToLogits, FlipLRMultiheadTTA, Flip4MultiheadTTA, \
    MultiscaleFlipLRMultiheadTTA
from retinopathy.train_utils import report_checkpoint


def download_from_s3(region='us-east-1', bucket="diabetic-retinopathy-data-from-radiology", s3_filename='test.png',
                     local_path="/opt/ml/code/"):
    s3_client = boto3.client('s3', region_name=region)
    try:
        s3_client.download_file(bucket, Key=s3_filename, Filename=local_path)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.info(f"The object s3://{bucket}/{s3_filename} in {region} does not exist.")
        else:
            raise


def image_with_name_in_dir(dirname, image_id):
    for ext in ['png', 'jpg', 'jpeg', 'tif']:
        image_path = path.join(dirname, f'{image_id}.{ext}')
        if path.isfile(image_path):
            return image_path
    raise FileNotFoundError(image_path)


def run_image_preprocessing(
        params,
        image_df: DataFrame,
        image_paths=None,
        preprocessing=None,
        image_size=None,
        crop_black=True,
        **kwargs) -> RetinopathyDataset:
    if image_paths is not None:
        if preprocessing is None:
            preprocessing = params.get('preprocessing', None)

        if image_size is None:
            image_size = params.get('image_size', 1024)
            image_size = (image_size, image_size)

        if 'diagnosis' in image_df:
            targets = image_df['diagnosis'].values
        else:
            targets = None

        return RetinopathyDataset(image_paths, targets, get_test_transform(image_size,
                                                                           preprocessing=preprocessing,
                                                                           crop_black=crop_black))


'''Not Changing variables'''
data_dir = '/opt/ml/code/'
model_name = 'seresnext50d_gap'
checkpoint_fname = 'last.pth'
checkpoint_path = path.join(data_dir, 'checkpoint', checkpoint_fname)

bucket = "diabetic-retinopathy-data-from-radiology"
images_dir = "/opt/ml/input/data"
image_size = 1024
params = {}
num_workers = multiprocessing.cpu_count()
CLASS_NAMES = []


# filename: inference.py

def model_fn(model_dir,
             tta=None,
             model_name=None,
             apply_softmax=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--need-features', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count(), help='')
    args = parser.parse_args()

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        # already available in this method torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['checkpoint_data']['cmd_args']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    report_checkpoint(checkpoint)

    if model_name is None:
        model_name = params['model']

    coarse_grading = params.get('coarse', False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    CLASS_NAMES = get_class_names(coarse_grading=coarse_grading)
    num_classes = len(CLASS_NAMES)
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    if apply_softmax:
        model = nn.Sequential(model, ApplySoftmaxToLogits())

    if tta == 'flip' or tta == 'fliplr':
        model = FlipLRMultiheadTTA(model)

    if tta == 'flip4':
        model = Flip4MultiheadTTA(model)

    if tta == 'fliplr_ms':
        model = MultiscaleFlipLRMultiheadTTA(model)

    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[id for id in range(torch.cuda.device_count())])
        model = model.eval()

    return model


def input_fn(request_body, content_type='application/json'):
    image_name = []

    if content_type == 'application/json':
        input_data = json.loads(request_body)
        region = input_data['region']

        for i in range(100):
            try:
                image_name.append(input_data[f'img{str(i)}'])
            except KeyError as e:
                print(e)
                break
        logger.info('Downloading the input diabetic retinopathy data.')

        image_df = DataFrame(image_name, columns=['id_code'])
        for id_code in image_df['id_code']:
            logger.info(f'Image filename: {id_code}')
            download_from_s3(region=region, bucket=bucket, s3_filename=id_code, local_path=images_dir)

        image_paths = image_df['id_code'].apply(lambda x: image_with_name_in_dir(images_dir, x))

        # Preprocessing the images
        dataset = run_image_preprocessing(
            params=params,
            apply_softmax=True,
            need_features=params['need_features'],
            image_df=image_df,
            image_paths=image_paths,
            batch_size=params['batch_size'],
            tta='fliplr',
            workers=num_workers,
            crop_black=True)

        return DataLoader(dataset, params['batch_size'],
                          pin_memory=True,
                          num_workers=num_workers)

    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')


def predict_fn(input_data, model, need_features=True):
    predictions = defaultdict(list)

    for batch in tqdm(input_data):
        input = batch['image'].cuda(non_blocking=True)
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
        outputs = model(input)

        predictions['image_id'].extend(batch['image_id'])
        if 'targets' in batch:
            predictions['diagnosis'].extend(to_numpy(batch['targets']).tolist())

        predictions['logits'].extend(to_numpy(outputs['logits']).tolist())
        predictions['regression'].extend(to_numpy(outputs['regression']).tolist())
        predictions['ordinal'].extend(to_numpy(outputs['ordinal']).tolist())
        if need_features:
            predictions['features'].extend(to_numpy(outputs['features']).tolist())

    del input_data
    return predictions


def output_fn(prediction_output, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(prediction_output), accept
    else:
        raise Exception(f'Requested unsupported ContentType in Accept:{accept}')

# if __name__ == '__main__':
#     main()
