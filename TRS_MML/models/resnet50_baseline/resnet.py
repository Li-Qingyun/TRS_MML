import torch
import logging
from typing import Dict
from torch import nn, Tensor
from mmf.models import BaseModel
from mmf.common.registry import registry
from torchvision.models.resnet import resnet50

logger = logging.getLogger(__name__)


@registry.register_model("resnet50")
class ResNet50(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/base/models/resnet50_baseline/defaults.yaml"

    def build(self):
        # build the CNN backbone
        self.resnet = resnet50(pretrained=self.config.base_args.pretrained)  #.get('pretrained', False))
        del self.resnet.fc

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
                    2048,
                    self.config.heads["hri_cls"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["hri_cls"] = hri_classifiers

        hsi_classifiers = nn.ModuleDict()
        for dataset_name in self.config.base_args.task_datasets.get("hsi_cls", []):
            hsi_classifiers[dataset_name] = nn.Sequential(
                nn.Linear(
                    2048,
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

        features = self.resnet.conv1(img_src)
        features = self.resnet.bn1(features)
        features = self.resnet.relu(features)
        features = self.resnet.maxpool(features)

        features = self.resnet.layer1(features)
        features = self.resnet.layer2(features)
        features = self.resnet.layer3(features)
        features = self.resnet.layer4(features)

        features = self.resnet.avgpool(features)
        resnet_output = {'hidden_states': torch.flatten(features, 1)}

        output = self.loss_calculation_fn[task_type](resnet_output, sample_list)
        return output

    def get_task_type(self, dataset_name):
        task_type = "detection"
        if dataset_name in self.config.heads.get('hri_cls', {}):
            task_type = "hri_cls"
        elif dataset_name in self.config.heads.get('hsi_cls', {}):
            task_type = "hsi_cls"
        return task_type
    
    def classifier_loss_calculation(self, resnet_outputs: Dict[str, Tensor], sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        hs = resnet_outputs["hidden_states"]
        if not self.config.loss_on_all_hs:
            hs = resnet_outputs["hidden_states"][-1:]
        losses = {}
        scores = None
        resnet_outputs = {}
        num_labels = self.config.heads[task_type][sample_list.dataset_name][
            "num_labels"
        ]

        for idx, current_hs in enumerate(hs):
            pooled_output = current_hs
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
                if sample_list.dataset_name == "vqa2":
                    loss *= sample_list.targets.size(1)
                losses[loss_prefix + f"loss_{idx}"] = loss

        resnet_outputs["scores"] = scores
        resnet_outputs["losses"] = losses
        return resnet_outputs

    def get_optimizer_parameters(self, config):
        input_proj_params = [
            {"params": self.hsi_lin_proj.parameters()}
        ]
        resnet_params = [
            {"params": self.resnet.conv1.parameters()},
            {"params": self.resnet.bn1.parameters()},
            {"params": self.resnet.relu.parameters()},
            {"params": self.resnet.maxpool.parameters()},
            {"params": self.resnet.layer1.parameters()},
            {"params": self.resnet.layer2.parameters()},
            {"params": self.resnet.layer3.parameters()},
            {"params": self.resnet.layer4.parameters()},
            {"params": self.resnet.avgpool.parameters()},
        ]
        output_head_params = [
            {"params": self.classifiers.parameters()}
        ]
        return input_proj_params + resnet_params + output_head_params

    def get_loss_fn(self, loss_type):
        if loss_type == "binary_cross_entropy_with_logits":
            return nn.functional.binary_cross_entropy_with_logits
        elif loss_type == "cross_entropy":
            return nn.functional.cross_entropy
        else:
            raise Exception(f"Unknown loss type: {loss_type}")