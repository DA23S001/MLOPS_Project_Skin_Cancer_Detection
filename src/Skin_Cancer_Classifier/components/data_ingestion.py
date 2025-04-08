import os, random
import zipfile
import gdown
import numpy as np
import pandas as pd
from sklearn import metrics

import sklearn
import albumentations
from PIL import Image


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from Skin_Cancer_Classifier import logger
from Skin_Cancer_Classifier.utils.common import get_size
from Skin_Cancer_Classifier.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


class Isic2019Raw(torch.utils.data.Dataset):
    """Pytorch dataset containing all the features, labels and datacenter
    information for Isic2019.

    Attributes
    ----------
    image_paths: list[str]
        the list with the path towards all features
    targets: list[int]
        the list with all classification labels for all features
    centers: list[int]
        the list for all datacenters for all features
    X_dtype: torch.dtype
        the dtype of the X features output
    y_dtype: torch.dtype
        the dtype of the y label output
    augmentations:
        image transform operations from the albumentations library,
        used for data augmentation
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.

    Parameters
    ----------
    X_dtype :
    y_dtype :
    augmentations :
    """

    def __init__(self, X_dtype=torch.float32, y_dtype=torch.int64, augmentations=None, data_path=None,):
        """
        Cf class docstring
        """

        if not (os.path.exists(data_path)):
            raise ValueError(f"The string {data_path} is not a valid path.")
        

        self.input_path = data_path

        self.dic = {"input_preprocessed": os.path.join(self.input_path, "ISIC_2019_Training_Input_preprocessed"),
                    "train_test_split": os.path.join(self.input_path, "train_test_split"),}
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        df2 = pd.read_csv(self.dic["train_test_split"])
        images = df2.image.tolist()
        self.image_paths = [os.path.join(self.dic["input_preprocessed"], image_name + ".jpg") for image_name in images ]
        self.targets = df2.target
        self.augmentations = augmentations
        self.centers = df2.center

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets[idx]

        # Image augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return (
            torch.tensor(image, dtype=self.X_dtype),
            torch.tensor(target, dtype=self.y_dtype),
        )


class FedIsic2019(Isic2019Raw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for the Isic2019 federated classification.
    One can instantiate this dataset with train or test data coming from either of
    the 6 centers it was created from or all data pooled.
    The train/test split is fixed and given in the train_test_split file.

    Parameters
    ----------
    center : int, optional
        Default to 0
    train : bool, optional
        Default to True
    pooled : bool, optional
        Default to False
    debug : bool, optional
        Default to False
    X_dtype : torch.dtype, optional
        Default to torch.float32
    y_dtype : torch.dtype, optional
        Default to torch.int64
    data_path: str
        If data_path is given it will ignore the config file and look for the
        dataset directly in data_path. Defaults to None.
    """

    def __init__(self, train: bool = True, X_dtype: torch.dtype = torch.float32, y_dtype: torch.dtype = torch.int64, data_path: str = None, centers: list = None):
        """Cf class docstring"""
        sz = 200
        if train:
            augmentations = albumentations.Compose(
                [
                    albumentations.RandomScale(0.07),
                    albumentations.Rotate(50),
                    albumentations.RandomBrightnessContrast(0.15, 0.1),
                    albumentations.Flip(p=0.5),
                    albumentations.Affine(shear=0.1),
                    albumentations.RandomCrop(sz, sz),
                    albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                    albumentations.Normalize(always_apply=True),
                ]
            )
        else:
            augmentations = albumentations.Compose(
                [
                    albumentations.CenterCrop(sz, sz),
                    albumentations.Normalize(always_apply=True),
                ]
            )

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            augmentations=augmentations,
            data_path=data_path,
        )

        self.centers_list = centers
        self.train_test = "train" if train else "test"
        df = pd.read_csv(self.dic["train_test_split"])

        df2 = df[(df['fold'] == self.train_test) & (df['center'].isin(self.centers_list))].reset_index(drop=True)

        images = df2.image.tolist()
        self.image_paths = [os.path.join(self.dic["input_preprocessed"], image_name + ".jpg") for image_name in images]
        self.targets = df2.target
        self.centers = df2.center



class BaselineLoss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    Attributes
    ----------
    alpha: torch.tensor of size 8, class weights
    gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
    the same as CE loss, increases gamma reduces the loss for the "hard to classify
    examples"
    """

    def __init__(
        self,
        alpha=torch.tensor(
            [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
        ),
        gamma=2.0,
    ):
        super(BaselineLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Weighted focal loss function
        Parameters
        ----------
        inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
        targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()
    

def acc_metric(y_true, logits):
    y_true = y_true.reshape(-1)
    preds = np.argmax(logits, axis=1)
    return metrics.balanced_accuracy_score(y_true, preds)
