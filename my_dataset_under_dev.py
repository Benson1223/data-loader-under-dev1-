from __future__ import division

import json
import logging

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
from sklearn.model_selection import StratifiedKFold
import os
import cv2
import csv
logger = logging.getLogger("global_logger")


def build_my_dataloader(cfg, stage, distributed=True):

    image_path = cfg["image_dir"]

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if stage == "train":
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    dataset = MyDataset(
        image_path,
        stage,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
    )

    # if distributed:
    #     sampler = DistributedSampler(dataset)
    # else:
    sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class MyDataset(BaseDataset):
    def __init__(
        self,
        image_path,
        stage,
        transform_fn,
        normalize_fn,
    ):
        self.root = image_path
        self.stage = stage
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.allfile = []
        self.filenames = []
        self.gt = []

        files = os.listdir(self.root)
        train_set = []
        val_set = []
        

        # Read datasets_label.csv and populate lists
        with open('../../../data_loader(under dev)/datasets_label.csv', mode='r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                file_path = os.path.join(self.root, row[0])
                if os.path.exists(file_path):
                    label = int(row[1] == 'FALSE')  # TRUE -> 0, FALSE -> 1
                    if row[2] == 'Training':
                        self.allfile.append(file_path)
                        self.gt.append(label)
                    elif row[2] == 'Testing' and self.stage == 'test':
                        self.filenames.append(file_path)
        
        # Split training data into train and validation sets using StratifiedKFold
        if self.stage in ['train', 'val']:
            skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
            for train_idx, val_idx in skf.split(self.allfile, self.gt):
                # Only use the first split
                if self.stage == "train":
                    for i in train_idx:
                        if self.gt[i]==0:
                            self.filenames.append(self.allfile[i])
                        
                elif self.stage == "val":
                    for j in val_idx:
                        self.filenames.append(self.allfile[j])
                        
                break

        print(f"{self.stage} set length: ", len(self.filenames))
                        

        ####ORI
        # for file in files:
        #     if file.endswith(".png"):
        #         self.allfile.append(os.path.join(self.root, file))
        #         self.gt.append(int(file.split("_")[0]))

        # skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        # for train_idx, val_idx in skf.split(self.allfile, self.gt):
            
        #     for i in train_idx:
        #         train_set.append(self.allfile[i])

        #     for j in val_idx:
        #         val_set.append(self.allfile[j])
                     
        #     break

        # if self.stage == "train":
        #     for file in train_set:
        #         if int(file.split("/")[-1].split("_")[0]) == 0:
        #             self.filenames.append(file)
        #     print("training set len: ", len(self.filenames))
        # elif self.stage == "val":
        #     self.filenames = val_set
        # else:
        #     self.root = os.path.join(self.root, "testing")
        #     test_files = os.listdir(self.root)
        #     for file in test_files:
        #         if file.endswith(".png"):
        #             self.filenames.append(os.path.join(self.root, file))
        #     print("testing lens: ", len(self.filenames))
   
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input = {}

        
        filename = self.filenames[idx].split("/")[-1]
        label = None
        with open('../../../data_loader(under dev)/datasets_label.csv', mode='r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if row[0]==filename:
                    label = 0 if row[1]=='TRUE' else 1
                    break
        if label is None:
            raise ValueError(f"Label for {filename} not found in CSV.")

        image = cv2.cvtColor(cv2.imread(self.filenames[idx]), cv2.COLOR_BGR2RGB)
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        #####ORI
        # filename = self.filenames[idx].split("/")[-1]
        # label = int(filename.split("_")[0])
        # image = cv2.cvtColor(cv2.imread(self.filenames[idx]), cv2.COLOR_BGR2RGB)
        # input.update(
        #     {
        #         "filename": filename,
        #         "height": image.shape[0],
        #         "width": image.shape[1],
        #         "label": label,
        #     }
        # )


        input["clsname"] = "t"

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if label == 0:  # good
            mask = np.zeros((image.height, image.width)).astype(np.uint8)
        elif label == 1:  # defective
            # path = self.filenames[idx].rsplit("/", 1)[0]
            # #mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            # mask = cv2.imread(os.path.join(path, "gt", filename), 0)
            mask = np.zeros((image.height, image.width)).astype(np.uint8)
        else:
            raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        # if self.colorjitter_fn:
        #     image = self.colorjitter_fn(image)

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.normalize_fn:
            image = self.normalize_fn(image)

        input.update({"image": image, "mask": mask})
        return input


