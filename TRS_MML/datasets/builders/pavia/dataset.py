# Copyright (c) Facebook, Inc. and its affiliates.
import os
import torch
import torchvision
import numpy as np
from torch.utils.data import TensorDataset
from mmf.common.sample import Sample, SampleList
from mmf.datasets.base_dataset import BaseDataset


class PaviaDataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "pavia"
        super().__init__(name, config, dataset_type, *args, **kwargs)
        self.dataset_name = name

        image_dir = self.config.images[self._dataset_type][imdb_file_index]
        self.image_dir = os.path.join(self.config.data_dir, image_dir)
        ann_dir = self.config.annotations[self._dataset_type][imdb_file_index]
        self.ann_dir = os.path.join(self.config.data_dir, ann_dir)

        imgs = torch.tensor(np.load(self.image_dir))
        anns = torch.tensor(np.load(self.ann_dir))

        self.dataset = TensorDataset(imgs, anns)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        transform_out = self.classification_RSI_processor(
            {"img": img, "target": target, "dataset_type": self._dataset_type}
        )
        img = transform_out["img"]
        target = transform_out["target"]

        current_sample = Sample()
        current_sample.image = img
        current_sample.targets = target

        return current_sample

    def __len__(self):
        return len(self.dataset)

    def prepare_batch(self, batch):
        # Should be a SampleList
        if not isinstance(batch, SampleList):
            # Try converting to SampleList
            batch = SampleList(batch)
        batch['targets'] = torch.LongTensor(batch['targets'])
        batch = batch.to(self._device)
        return batch