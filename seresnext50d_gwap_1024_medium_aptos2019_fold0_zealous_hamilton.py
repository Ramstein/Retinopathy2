hyperparameters = {
    "seed": 42,
    #   "fast": false,
    #   "mixup": false,
    #   "balance": false,
    #   "balance_datasets": false,
    #   "swa": false,
    #   "show": false,
    #   "use_idrid": false,
    #   "use_messidor": false,
    #   "use_aptos2015": false,
    # "use_aptos2019": "",
    # "verbose": "",
    #   "coarse": false,
    "accumulation-steps": 1,
    "data-dir": "/opt/ml/input/data",  # in SageMaker instance data-dir = '/home/ec2-user/SageMaker/data/'
    "model": "seresnext50d_gwap",
    "batch-size": 32,
    "epochs": 200,
    "early-stopping": 10,
    # "fold": [
    #     0,
    #     1,
    #     2,
    #     3
    # ],
    #   "freeze_encoder": false,
    "learning-rate": 0.0001,
    # "criterion_reg": [
    #     "mse"
    # ],
    #   "criterion_ord": null,
    # "criterion_cls": [
    #     "focal_kappa"
    # ],
    "l1": 0.0002,
    # "l2": 0,
    "optimizer": "AdamW",
    #   "preprocessing": null,
    #   "checkpoint": null,
    #   "workers": 8, # default to multiprocessing.cpu_count()
    "augmentations": "medium",
    #   "tta": null,
      "transfer": 'se_resnext50_32x4d-a260b3a4.pth',
    #   "fp16": true,
    "scheduler": "multistep",
    "size": 1024,
    "weight-decay": 0.0001,
    #   "weight_decay_step": null,
    "dropout": 0.2,
    # "warmup": 0,
    #   "experiment": null
}
