# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/models/backbone.py
"""
Backbone modules.
"""
import math
import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from mmf.models.unit.misc import NestedTensor
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


logger = logging.getLogger(__name__)


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        return_interm_layers: bool,
    ):
        super().__init__()
        if 'resnet' in self.name:
            for name, parameter in backbone.named_parameters():
                if (
                    not train_backbone
                    or "layer2" not in name
                    and "layer3" not in name
                    and "layer4" not in name
                ):
                    parameter.requires_grad_(False)
            if return_interm_layers:  # TODO: Migrate to the configuration
                # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                if self.name in ['resnet18', 'resnet34']:
                    self.num_channels = [128, 256, 512]
                else:
                    self.num_channels = [512, 1024, 2048]
            else:
                return_layers = {"layer4": 0}
                self.num_channels = [512] if self.name in ['resnet18', 'resnet34'] else [2048]
        elif 'vgg' in self.name:
            return_layers = {"features": 0}
            self.num_channels = [512]
            assert not return_interm_layers  # Only support single-scale for VGG
        else:
            raise NotImplementedError(
                f"The return_layers and num_channels of {self.name} requires manual setting."
            )
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        self.name = name

        backbone_kwargs = {}
        if 'resnet' in name:
            backbone_kwargs['replace_stride_with_dilation'] = [False, False, dilation]
            backbone_kwargs['norm_layer'] = FrozenBatchNorm2d
        backbone_kwargs['pretrained'] = True

        backbone = getattr(torchvision.models, name)(**backbone_kwargs)

        if backbone_kwargs['pretrained'] == True :
            logger.info(f"Loaded imagenet pretrained '{name}' backbone.")

        super().__init__(backbone, train_backbone, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self,
                 backbone: nn.Module,
                 neck: nn.Module,
                 position_embedding: nn.Module
                 ):
        super().__init__(backbone, neck, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        xs = self[1](xs, tensor_list)
        out = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[2](x).to(x.tensors.dtype))

        return out, pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Neck(nn.Module):
    def __init__(self, in_channels, out_channels, num_feature_levels = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(in_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels[_], out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels[-1], out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[-1], out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
            )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, in_features: OrderedDict, tensor_list = None):
        out = OrderedDict()
        for l, (name, feat) in enumerate(in_features.items()):
            src, mask = feat.decompose()
            src = self.input_proj[l](src)
            out[name] = NestedTensor(src, mask)
        if self.num_feature_levels > len(in_features):
            _len_in_features = len(in_features)
            for l in range(_len_in_features, self.num_feature_levels):
                if l == _len_in_features:
                    src = self.input_proj[l](in_features[next(reversed(in_features))].tensors)
                else:
                    src = self.input_proj[l](out[next(reversed(out))].tensors)
                if tensor_list is not None:
                    mask = F.interpolate(
                        tensor_list.mask[None].float(), size=src.shape[-2:]
                    ).to(torch.bool)[0]
                else:
                    mask = F.interpolate(
                        out[next(reversed(out))].mask.float(), size=src.shape[-2:]
                    ).to(torch.bool)[0]
                out[f'{l}'] = NestedTensor(src, mask)
        return out


def build_unit_convnet_backbone(args):
    position_embedding = PositionEmbeddingSine(
        args.encoder_hidden_dim // 2, normalize=True
    )
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.num_feature_levels > 1
    backbone = Backbone(
        args.backbone, train_backbone, return_interm_layers, args.dilation
    )
    neck = Neck(
        backbone.num_channels, args.decoder_hidden_dim, args.num_feature_levels
    )
    model = Joiner(backbone, neck, position_embedding)
    return model
