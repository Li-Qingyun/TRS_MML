# Copyright (c) Facebook, Inc. and its affiliates.
from .dataset import RESISCDataset
from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("resisc")
class RESISCBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="resisc", dataset_class=RESISCDataset
        )

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/datasets/resisc/defaults.yaml"
