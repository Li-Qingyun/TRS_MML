# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import gather_tensor_along_batch, object_to_byte_tensor
from torch import Tensor, nn


class RESISCDataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "resisc"
        super().__init__(name, config, dataset_type, *args, **kwargs)
        self.dataset_name = name

        image_dir = self.config.images[self._dataset_type][imdb_file_index]
        self.image_dir = os.path.join(self.config.data_dir, image_dir)

        img_transforms = transforms.Compose([
            transforms.ToTensor()       # TODO: Normalization ?
        ])

        self.dataset = torchvision.datasets.ImageFolder(
            self.image_dir, transform=img_transforms
        )

    def __getitem__(self, idx):
        current_sample = Sample()
        current_sample.image, current_sample.target = self.dataset[idx]
        return current_sample

    def __len__(self):
        return len(self.dataset)

