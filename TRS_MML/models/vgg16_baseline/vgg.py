import torch
import logging
from typing import Dict
from torch import nn, Tensor
from mmf.models import BaseModel
from mmf.common.registry import registry
from torchvision.models.vgg import vgg16_bn

logger = logging.getLogger(__name__)


@registry.register_model("vgg16")
class VGG16(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/base/models/vgg16_baseline/defaults.yaml"

    def build(self):
        # build the CNN backbone
        self.vgg = vgg16_bn(pretrained=self.config.base_args.pretrained)  #.get('pretrained', False))
        del self.vgg.classifier[-1]

        # build the hsi linear projection
        hsi_lin_proj = nn.ModuleDict()
        for dataset_name in self.config.base_args.task_datasets.get("hsi_cls", []):
            hsi_lin_proj[dataset_name] = nn.Conv2d(
                in_channels=self.config.heads["hsi_cls"][dataset_name]["num_channels"],
                out_channels=3, kernel_size=1,
            )
        self.hsi_lin_proj = hsi_lin_proj

        # build the task-specific output heads for Classification on Optical RSI (hri) or HSI
        self.classifiers = nn.ModuleDict()

        hri_classifiers = nn.ModuleDict()
        for dataset_name in self.config.base_args.task_datasets.get("hri_cls", []):
            hri_classifiers[dataset_name] = nn.Sequential(
                nn.Linear(
                    4096,
                    self.config.heads["hri_cls"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["hri_cls"] = hri_classifiers

        hsi_classifiers = nn.ModuleDict()
        for dataset_name in self.config.base_args.task_datasets.get("hsi_cls", []):
            hsi_classifiers[dataset_name] = nn.Sequential(
                nn.Linear(
                    4096,
                    self.config.heads["hsi_cls"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["hsi_cls"] = hsi_classifiers

        self.dropout = nn.Dropout(self.config.base_args.dropout_config)

        # build loss calculation func
        self.loss_calculation_fn = {}
        self.loss_calculation_fn["hri_cls"] = self.classifier_loss_calculation
        self.loss_calculation_fn["hsi_cls"] = self.classifier_loss_calculation

        self.losses_dict = {}
        self.losses_dict["hri_cls"] = {
            name: self.get_loss_fn(self.config.heads["hri_cls"][name]["loss_type"])
            for name in self.config.heads.get('hri_cls', {})
        }
        self.losses_dict["hsi_cls"] = {
            name: self.get_loss_fn(self.config.heads["hsi_cls"][name]["loss_type"])
            for name in self.config.heads.get('hsi_cls', {})
        }

    def forward(self, sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        img_src = sample_list.image

        if task_type == 'hsi_cls':
            img_src = self.hsi_lin_proj[sample_list.dataset_name](img_src)

        features = self.vgg(img_src)
        vgg_output = {'hidden_states': torch.flatten(features, 1)}

        output = self.loss_calculation_fn[task_type](vgg_output, sample_list)
        return output

    def get_task_type(self, dataset_name):
        task_type = "detection"
        if dataset_name in self.config.heads.get('hri_cls', {}):
            task_type = "hri_cls"
        elif dataset_name in self.config.heads.get('hsi_cls', {}):
            task_type = "hsi_cls"
        return task_type
    
    def classifier_loss_calculation(self, vgg_outputs: Dict[str, Tensor], sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        hs = vgg_outputs["hidden_states"]
        losses = {}
        scores = None
        vgg_outputs = {}
        num_labels = self.config.heads[task_type][sample_list.dataset_name][
            "num_labels"
        ]

        pooled_output = hs
        pooled_output = self.dropout(pooled_output)
        logits = self.classifiers[task_type][sample_list.dataset_name](
            pooled_output
        )
        reshaped_logits = logits.contiguous().view(-1, num_labels)
        scores = reshaped_logits
        # skip loss computation on test set (which usually doesn't contain labels)
        if sample_list.dataset_type != "test":
            loss_prefix = f"{sample_list.dataset_type}/{sample_list.dataset_name}/"
            loss = self.losses_dict[task_type][sample_list.dataset_name](
                scores, sample_list.targets
            )
            losses[loss_prefix + f"loss_00"] = loss

        vgg_outputs["scores"] = scores
        vgg_outputs["losses"] = losses
        return vgg_outputs

    def get_optimizer_parameters(self, config):
        input_proj_params = [
            {"params": self.hsi_lin_proj.parameters()}
        ]
        vgg_params = [
            {"params": self.vgg.parameters()},
        ]
        output_head_params = [
            {"params": self.classifiers.parameters()}
        ]
        return input_proj_params + vgg_params + output_head_params

    def get_loss_fn(self, loss_type):
        if loss_type == "binary_cross_entropy_with_logits":
            return nn.functional.binary_cross_entropy_with_logits
        elif loss_type == "cross_entropy":
            return nn.functional.cross_entropy
        else:
            raise Exception(f"Unknown loss type: {loss_type}")