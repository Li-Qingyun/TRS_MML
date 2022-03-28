# Copyright (c) Facebook, Inc. and its affiliates.

from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from mmf.common.registry import registry
from mmf.modules.optimizers import AdamW


registry.register_optimizer("sgd")(SGD)
registry.register_optimizer("adam")(Adam)