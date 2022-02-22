# Copyright (c) Facebook, Inc. and its affiliates.
from .dataset import IndianDataset
from mmf.common.registry import registry
from mmf.datasets.concat_dataset import MMFConcatDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("indian")
class IndianBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="indian", dataset_class=IndianDataset
        )

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/datasets/indian/defaults.yaml"

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