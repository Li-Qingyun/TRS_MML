# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from torch.optim.sgd import SGD
from torch.optim.adam import Adam


registry.register_optimizer("sgd")(SGD)
registry.register_optimizer("adam")(Adam)