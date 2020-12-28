import json
import torch
import tarfile
import pickle
import matplotlib.pyplot as plt
import torchvision as tv
import pathlib                          # Path management tool (standard library)
import subprocess                       # Runs shell commands via Python (standard library)
import sagemaker                        # SageMaker Python SDK
from sagemaker.pytorch import PyTorch   # PyTorch Estimator for TensorFlow