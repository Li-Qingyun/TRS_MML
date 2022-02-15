# Copyright (c) Facebook, Inc. and its affiliates.

from bisect import bisect_right

from mmf.common.registry import registry
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import (
    get_constant_schedule_with_warmup
)


@registry.register_scheduler("warmup_constant")
class WarmupConstantScheduler(LambdaLR):
    def __new__(cls, optimizer, *args, **kwargs):
        return get_constant_schedule_with_warmup(optimizer, *args, **kwargs)




