import logging
import os
from typing import Dict, Optional, Tuple

import torch
from mmf.datasets.processors.processors import Processor
from mmf.utils.configuration import get_global_config
from mmf.utils.distributed import is_xla
from mmf.utils.build import _add_extra_args_for_dataloader
from omegaconf import DictConfig
from packaging import version


try:
    import torch_xla.core.xla_model as xm  # noqa
    import torch_xla.distributed.parallel_loader as xla_pl  # noqa
except ImportError:
    xm = None

ProcessorDict = Dict[str, Processor]
logger = logging.getLogger(__name__)


def build_dataloader_and_sampler(
    dataset_instance: torch.utils.data.Dataset, datamodule_config: DictConfig
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.Sampler]]:
    """Builds and returns a dataloader along with its sample

    Args:
        dataset_instance (torch.utils.data.Dataset): Instance of dataset for which
            dataloader has to be created
        datamodule_config (omegaconf.DictConfig): Datamodule configuration; required
            for infering params for dataloader

    Returns:
        Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.Sampler]]:
            Tuple of Dataloader and Sampler instance
    """
    from mmf.common.batch_collator import BatchCollator

    training_config = get_global_config("training")
    # Support params coming in from dataloader params
    other_args = {
        "num_workers": datamodule_config.get(
            "num_workers", training_config.get("num_workers", 4)
        ),
        "pin_memory": datamodule_config.get(
            "pin_memory", training_config.get("pin_memory", False)
        ),
        "shuffle": datamodule_config.get("shuffle", None),
        "batch_size": datamodule_config.get("batch_size", None),
    }
    dataset_type = dataset_instance.dataset_type
    if (dataset_instance.dataset_type in ['val', 'test']) \
        and datamodule_config.get("specific_val_batch_size", False):
        val_batch_size = datamodule_config.get("val_batch_size", 0)
        assert val_batch_size > 0
        logger.info(
            "Batch size of {}_dataloader is set {}".format(dataset_type, val_batch_size)
        )
        other_args["batch_size"] = val_batch_size
    if version.parse(torch.__version__) >= version.parse("1.8"):
        # only use persistent workers in PyTorch 1.8 or higher
        # (PyTorch 1.7 also has this option but doesn't support it correctly due to
        # https://github.com/pytorch/pytorch/issues/48370)
        other_args["persistent_workers"] = (
            datamodule_config.get(
                "persistent_workers", training_config.get("persistent_workers", True)
            ),
        )
        if other_args["persistent_workers"] and other_args["num_workers"] == 0:
            logger.warning(
                "persistent_workers cannot be used together with num_workers == 0; "
                "setting persistent_workers to False"
            )
            other_args["persistent_workers"] = False

    # IterableDataset returns batches directly, so no need to add Sampler
    # or batch size as user is expected to control those. This is a fine
    # assumption for now to not support single item based IterableDataset
    # as it will add unnecessary complexity and config parameters
    # to the codebase
    if not isinstance(dataset_instance, torch.utils.data.IterableDataset):
        other_args = _add_extra_args_for_dataloader(dataset_instance, other_args)
    else:
        other_args.pop("shuffle")

    # Set drop_last=True when using XLA to have constant batch size.
    # In this case we also need to set drop_last=True in DistributedSampler.
    loader = torch.utils.data.DataLoader(
        dataset=dataset_instance,
        collate_fn=BatchCollator(
            dataset_instance.dataset_name, dataset_instance.dataset_type
        ),
        drop_last=is_xla(),  # see also MultiDatasetLoader.__len__
        **other_args,
    )

    if is_xla():
        device = xm.xla_device()
        loader = xla_pl.MpDeviceLoader(loader, device)

    if other_args["num_workers"] >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    loader.dataset_type = dataset_instance.dataset_type

    return loader, other_args.get("sampler", None)