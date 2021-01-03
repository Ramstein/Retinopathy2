import argparse
import multiprocessing
import os
import time

import boto3
import pandas as pd
import torch
from botocore.exceptions import ClientError
from pytorch_toolbelt.utils import fs

from retinopathy.inference import run_model_inference


def download_from_s3(s3_filename, local_path="test"):
    bucket = "diabetic-retinopathy-data-from-radiology"
    region_name = "us-east-1"

    s3_client = boto3.client('s3', region_name=region_name)
    # print("Downloading file {} to {}".format(s3_filename, local_path))
    try:
        s3_client.download_file(bucket, Key=s3_filename, Filename=local_path)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def image_with_name_in_dir(dirname, image_id):
    for ext in ['png', 'jpg', 'jpeg', 'tif']:
        image_path = os.path.join(dirname, f'{image_id}.{ext}')
        if os.path.isfile(image_path):
            return image_path
    raise FileNotFoundError(image_path)




# filename: inference.py
# def input_fn(request_body, request_content_type)
# def predict_fn(input_data, model)
# def output_fn(prediction, content_type)

def model_fn(model_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--need-features', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=multiprocessing.cpu_count(),
                        help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-w', '--workers', type=int, default=4, help='')

    args = parser.parse_args()
    need_features = args.need_features = True
    batch_size = args.batch_size = 32
    num_workers = args.workers = multiprocessing.cpu_count()

    # checkpoint_fname = args.input  # pass just single checkpoint filename as arg
    '''Not Changing variables'''
    data_dir = '/opt/ml/code/'
    checkpoint_fname = ''
    checkpoint_path = os.path.join(data_dir, 'checkpoint', checkpoint_fname)
    current_milli_time = lambda: str(round(time.time() * 1000))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    params = checkpoint['checkpoint_data']['cmd_args']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if workers is None:
        workers = multiprocessing.cpu_count()

    report_checkpoint(checkpoint)

    if model_name is None:
        model_name = params['model']

    if batch_size is None:
        batch_size = params.get('batch_size', 1)

    coarse_grading = params.get('coarse', False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_classes = len(get_class_names(coarse_grading=coarse_grading))
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    if apply_softmax:
        model = nn.Sequential(model, ApplySoftmaxToLogits())

    if tta == 'flip' or tta == 'fliplr':
        model = FlipLRMultiheadTTA(model)

    if tta == 'flip4':
        model = Flip4MultiheadTTA(model)


    logger.info('Loading the model.')
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 10),
        nn.Dropout(0.4),
        nn.LogSoftmax(dim=1))

    with open(os.path.join(model_dir, 'model_0.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    logger.info('Done loading model')
    return model








    '''Not Changing variables'''
    data_dir = '/opt/ml/input/data'
    checkpoint_path = os.path.join(data_dir, 'model', checkpoint_fname)
    current_milli_time = lambda: str(round(time.time() * 1000))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    params = checkpoint['checkpoint_data']['cmd_args']

    # Make OOF predictions
    images_dir = os.path.join(data_dir, "ratinopathy", current_milli_time())

    retino = pd.read_csv(os.path.join(data_dir, 'aptos-2019', 'test.csv'))
    '''Downloading fundus photography files'''
    for id_code in retino['id_code']:
        download_from_s3(s3_filename="aptos-2019/train.csv", local_path=os.path.join(images_dir, id_code))

    image_paths = retino['id_code'].apply(lambda x: image_with_name_in_dir(images_dir, x))

    # Now run inference on Aptos2019 public test, will return a pd.DataFrame having image_id, logits, regrssions, ordinal, features
    ratinopathy = run_model_inference(checkpoint=checkpoint,
                                      params=params,
                                      apply_softmax=True,
                                      need_features=need_features,
                                      retino=retino,
                                      image_paths=image_paths,
                                      batch_size=batch_size,
                                      tta='fliplr',
                                      workers=num_workers,
                                      crop_black=True)
    ratinopathy.to_pickle(fs.change_extension(checkpoint_fname, '_ratinopathy_predictions.pkl'))








if __name__ == '__main__':
    main()
