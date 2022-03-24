import torch
import logging
from typing import Dict
from torch import nn, Tensor
from mmf.models.unit import UniT
from mmf.models.unit.unit_base_model import (
    MLP,
    AttributeHead,
    build_detection_loss,
)
from mmf.common.registry import registry
from TRS_MML.models.trs.trs_base_model import TRSBaseModel


logger = logging.getLogger(__name__)


@registry.register_model("trs")
class TRS(UniT):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/base/models/trs/defaults.yaml"

    def build(self):
        # build the base model (based on DETR)
        self.base_model = TRSBaseModel(self.config)

        # def keep_only_backbone_params(model_state_dict):
        #     keys = list(model_state_dict.keys())
        #     for k in keys:
        #         if "backbone" not in k:
        #             model_state_dict.pop(k)
        #
        # ckpt_path = self.config.base_ckpt_path
        # if ckpt_path != "" and ('resnet50' in self.config.base_args.backbone):
        #     logger.info(f"initializing base model (UniT) from {ckpt_path}")
        #     if ckpt_path.startswith("https"):
        #         base_checkpoint = torch.hub.load_state_dict_from_url(
        #             ckpt_path, check_hash=True
        #         )
        #     else:
        #         base_checkpoint = torch.load(ckpt_path)
        #     if self.config.base_ckpt_load_backbone_only:
        #         keep_only_backbone_params(base_checkpoint["model"])
        #         self.base_model.load_state_dict(
        #             base_checkpoint["model"], strict=False
        #         )
        #     else:
        #         self.base_model.load_state_dict(
        #             base_checkpoint["model"], strict=True
        #         )

        # build the task-specific output heads for Object Detection on Optical RSI
        self.class_embeds = nn.ModuleDict()
        self.bbox_embeds = nn.ModuleDict()
        self.det_losses = nn.ModuleDict()

        detr_hidden_dim = self.config.base_args.decoder_hidden_dim

        for dataset_name in self.config.base_args.num_queries.get("detection", []):
            num_cls = self.config.heads["detection"][dataset_name]["num_classes"]
            self.class_embeds[dataset_name] = nn.Linear(detr_hidden_dim, num_cls + 1)
            self.bbox_embeds[dataset_name] = MLP(detr_hidden_dim, detr_hidden_dim, 4, 3)
            attr_head = None
            if self.config.heads["detection"][dataset_name]["use_attr"]:
                attr_head = AttributeHead(
                    num_cls, self.config.base_args.attribute_class_num, detr_hidden_dim
                )
            self.det_losses[dataset_name] = build_detection_loss(
                self.config.base_args, num_cls, attr_head
            )

        # build the task-specific output heads for Classification on Optical RSI (hri) or HSI
        self.classifiers = nn.ModuleDict()

        hri_classifiers = nn.ModuleDict()
        for dataset_name in self.config.base_args.num_queries.get("hri_cls", []):
            hri_classifiers[dataset_name] = nn.Sequential(
                nn.Linear(
                    detr_hidden_dim,
                    self.config.heads["hri_cls"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["hri_cls"] = hri_classifiers

        hsi_classifiers = nn.ModuleDict()
        for dataset_name in self.config.base_args.num_queries.get("hsi_cls", []):
            hsi_classifiers[dataset_name] = nn.Sequential(
                nn.Linear(
                    detr_hidden_dim,
                    self.config.heads["hsi_cls"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["hsi_cls"] = hsi_classifiers

        self.dropout = nn.Dropout(self.config.base_args.dropout_config)

        # build loss calculation func
        self.loss_calculation_fn = {}
        self.loss_calculation_fn["detection"] = self.detection_loss_calculation
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

        detr_outputs = self.base_model(
            img_src=img_src,
            task_type=task_type,
            dataset_name=sample_list.dataset_name,
            task_idx=self.get_task_idx(sample_list.dataset_name),
        )

        output = self.loss_calculation_fn[task_type](detr_outputs, sample_list)
        return output

    def get_task_type(self, dataset_name):
        task_type = "detection"
        if dataset_name in self.config.heads.get('hri_cls', {}):
            task_type = "hri_cls"
        elif dataset_name in self.config.heads.get('hsi_cls', {}):
            task_type = "hsi_cls"
        return task_type

    def classifier_loss_calculation(self, detr_outputs: Dict[str, Tensor], sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        hs = detr_outputs["hidden_states"]
        if not self.config.loss_on_all_hs:
            hs = detr_outputs["hidden_states"][-1:]
        num_queries = self.config.base_args.num_queries[task_type][
            sample_list.dataset_name
        ]
        assert hs[0].size(1) == num_queries
        losses = {}
        scores = None
        detr_outputs = {}
        num_labels = self.config.heads[task_type][sample_list.dataset_name][
            "num_labels"
        ]

        for idx, current_hs in enumerate(hs):
            pooled_output = current_hs[:, -num_queries, :]
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

        detr_outputs["scores"] = scores
        detr_outputs["losses"] = losses
        return detr_outputs

    def get_optimizer_parameters(self, config):
        detr_params = [
            {
                "params": [
                    p
                    for n, p in self.base_model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.base_model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.config.base_args.lr_backbone,
            },
        ]

        other_params = [
            {"params": self.classifiers.parameters()},
            {"params": self.class_embeds.parameters()},
            {"params": self.bbox_embeds.parameters()},
            {"params": self.det_losses.parameters()},
        ]

        return detr_params + other_params