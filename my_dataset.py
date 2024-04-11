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


        for file in files:
            if file.endswith(".png"):
                self.allfile.append(os.path.join(self.root, file))
                self.gt.append(int(file.split("_")[0]))

        skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        for train_idx, val_idx in skf.split(self.allfile, self.gt):
            
            for i in train_idx:
                train_set.append(self.allfile[i])

            for j in val_idx:
                val_set.append(self.allfile[j])
                     
            break

        if self.stage == "train":
            for file in train_set:
                if int(file.split("/")[-1].split("_")[0]) == 0:
                    self.filenames.append(file)
            print("training set len: ", len(self.filenames))
        elif self.stage == "val":
            self.filenames = val_set
        else:
            self.root = os.path.join(self.root, "testing")
            test_files = os.listdir(self.root)
            for file in test_files:
                if file.endswith(".png"):
                    self.filenames.append(os.path.join(self.root, file))
            print("testing lens: ", len(self.filenames))
   
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input = {}

        # read image
        filename = self.filenames[idx].split("/")[-1]
        label = int(filename.split("_")[0])
        image = cv2.cvtColor(cv2.imread(self.filenames[idx]), cv2.COLOR_BGR2RGB)
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )


        input["clsname"] = "tsmc"

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if label == 0:  # good
            mask = np.zeros((image.height, image.width)).astype(np.uint8)
        elif label == 1:  # defective
            path = self.filenames[idx].rsplit("/", 1)[0]
            #mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            mask = cv2.imread(os.path.join(path, "gt", filename), 0)
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


