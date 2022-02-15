# Copyright (c) Facebook, Inc. and its affiliates.
from .dataset import DIORDataset
from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("dior")
class DIORBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="dior", dataset_class=DIORDataset
        )

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/datasets/dior/defaults.yaml"
