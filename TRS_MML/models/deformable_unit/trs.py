import math
import copy
import torch
import logging
from typing import Dict
from torch import nn, Tensor
from mmf.models.unit import UniT
from TRS_MML.models.deformable_unit.unit_base_model import (
    MLP,
    build_detection_loss,
)
from mmf.common.registry import registry
from mmf.utils.distributed import byte_tensor_to_object
from TRS_MML.models.deformable_unit.unit_base_model import UniTBaseModel
from TRS_MML.models.deformable_unit.misc import inverse_sigmoid


logger = logging.getLogger(__name__)


@registry.register_model("deformable_trs")
class DTRS(UniT):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/base/models/trs/develop.yaml"

    def build(self):
        self.two_stage = True  # TODO: ADD API
        self.with_box_refine = True  # TODO: ADD API
        two_stage = self.two_stage  # TODO: ADD API
        with_box_refine = self.with_box_refine  # TODO: ADD API
        self.config.base_args.two_stage = self.two_stage  # TODO: ADD API
        self.config.base_args.with_box_refine = self.with_box_refine  # TODO: ADD API

        # build the base model (based on DETR)
        self.base_model = UniTBaseModel(self.config.base_args)

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
        detr_num_layers = self.config.base_args.dec_layers

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        for dataset_name in self.config.base_args.num_queries.get("detection", []):
            num_cls = self.config.heads["detection"][dataset_name]["num_classes"]
            class_embed = nn.Linear(detr_hidden_dim, num_cls + 1)
            bbox_embed = MLP(detr_hidden_dim, detr_hidden_dim, 4, 3)
            det_losses = build_detection_loss(self.config.base_args, num_cls)
            # init
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            class_embed.bias.data = torch.ones(num_cls + 1) * bias_value
            nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
            # if two-stage, the last class_embed and bbox_embed is for region proposal generation
            num_pred = (detr_num_layers + 1) if two_stage else detr_num_layers
            if with_box_refine:
                class_embed = _get_clones(class_embed, num_pred)
                bbox_embed = _get_clones(bbox_embed, num_pred)
                nn.init.constant_(bbox_embed[0].layers[-1].bias.data[2:], -2.0)
                # hack implementation for iterative bounding box refinement
                self.base_model.transformer.decoder.bbox_embed = bbox_embed  # TODO: for which dataset?
            else:
                nn.init.constant_(bbox_embed.layers[-1].bias.data[2:], -2.0)
                class_embed = nn.ModuleList([class_embed for _ in range(num_pred)])
                bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_pred)])
                self.base_model.transformer.decoder.bbox_embed = None  # TODO: for which dataset?
            if two_stage:
                # hack implementation for two-stage
                self.base_model.transformer.decoder.class_embed = class_embed  # TODO: for which dataset?
                for embed in bbox_embed:
                    nn.init.constant_(embed.layers[-1].bias.data[2:], 0.0)
            self.class_embeds[dataset_name] = class_embed
            self.bbox_embeds[dataset_name] = bbox_embed
            self.det_losses[dataset_name] = det_losses

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

        detr_outputs = self.base_model(
            samples=sample_list.image,
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

    def detection_loss_calculation(self, detr_outputs: Dict[str, Tensor], sample_list):
        hs = detr_outputs["hidden_states"]
        init_reference = detr_outputs["init_reference"]
        inter_references = detr_outputs["inter_references"]

        outputs_classes = []
        outputs_coords = []
        dataset_name = sample_list.dataset_name
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embeds[dataset_name][lvl](hs[lvl])
            tmp = self.bbox_embeds[dataset_name][lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        detr_outputs.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
        )

        # skip loss computation on test set (which usually doesn't contain labels)
        if sample_list.dataset_type != "test":
            if self.two_stage:
                enc_outputs_coord_unact = detr_outputs['enc_outputs_coord_unact']
                enc_outputs_class = detr_outputs['enc_outputs_class']
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                detr_outputs['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

            if self.config.base_args.aux_loss:
                detr_outputs["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

            criterion = self.det_losses[sample_list.dataset_name]
            targets = [byte_tensor_to_object(t) for t in sample_list.targets_enc]
            targets = [{k: v.to(hs.device) for k, v in t.items()} for t in targets]
            sample_list.targets = targets
            loss_dict = criterion(detr_outputs, sample_list.targets)
            weight_dict = criterion.weight_dict
            loss_prefix = f"{sample_list.dataset_type}/{sample_list.dataset_name}/"
            losses = {
                (loss_prefix + f"{k}"): loss_dict[k]
                * weight_dict[k]
                * self.config.detection_loss_weight
                for k in loss_dict.keys()
                if k in weight_dict
            }
            detr_outputs["losses"] = losses

        return detr_outputs

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
        backbone_params = []
        sampling_offsets_params = []
        reference_points_params = []
        transformers_params = []
        for n, p in self.base_model.named_parameters():
            if p.requires_grad:
                if "backbone" in n :
                    backbone_params.append(p)
                elif "sampling_offsets" in n :
                    sampling_offsets_params.append(p)
                elif "reference_points" in n :
                    reference_points_params.append(p)
                else:
                    transformers_params.append(p)

        detr_params = [
            {
                "params": transformers_params
            },
            {
                "params": backbone_params,
                "lr_mult": self.config.base_args.lr_mult.backbone,
            },
            {
                "params": sampling_offsets_params,
                "lr_mult": self.config.base_args.lr_mult.sampling_offsets,
            },
            {
                "params": reference_points_params,
                "lr_mult": self.config.base_args.lr_mult.reference_points,
            },
        ]

        other_params = [
            {"params": self.classifiers.parameters()},
            {"params": self.det_losses.parameters()},
        ]

        if not self.with_box_refine:
            other_params.append({"params": self.bbox_embeds.parameters()})
        if not self.two_stage:
            other_params.append({"params": self.class_embeds.parameters()})

        return detr_params + other_params