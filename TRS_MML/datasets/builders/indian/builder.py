# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Optional
from torch.utils.data import Dataset
from mmf.common.registry import registry
from mmf.datasets.concat_dataset import MMFConcatDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from TRS_MML.utils.build import build_dataloader_and_sampler
from TRS_MML.datasets.builders.indian.dataset import IndianDataset


@registry.register_builder("indian")
class IndianBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="indian", dataset_class=IndianDataset
        )

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/base/datasets/indian/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        self.config = config

        split_dataset_from_train = self.config.get("split_train", False)
        if split_dataset_from_train:
            config = self._modify_dataset_config_for_split(config)

        dataset_class = self.dataset_class
        dataset = dataset_class(config, dataset_type, 0)
        datasets= [dataset]

        dataset = MMFConcatDataset(datasets)
        if split_dataset_from_train:
            dataset = self._split_dataset_from_train(dataset, dataset_type)

        self.dataset = dataset
        return self.dataset

    def build_dataloader(
        self, dataset_instance: Optional[Dataset], dataset_type: str, *args, **kwargs
    ):
        if dataset_instance is None:
            raise TypeError(
                f"dataset instance for {dataset_type} hasn't been set and is None"
            )
        dataset_instance.dataset_type = dataset_type
        dataloader, _ = build_dataloader_and_sampler(dataset_instance, self.config)
        return dataloader