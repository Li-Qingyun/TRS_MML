from typing import Optional
from torch import Tensor, nn
from omegaconf import DictConfig
from mmf.models.unit.misc import NestedTensor
from TRS_MML.models.components.backbone import build_unit_convnet_backbone
from TRS_MML.models.trs.trs_transformer import TRSTransformer, TRSDeformableTransformer


class TRSBaseModel(nn.Module):
    def __init__(self, config):
        args = config.base_args
        self.use_deformable_attention = args.use_deformable_attention
        self.multi_scale = args.multi_scale
        self.two_stage = args.two_stage
        super().__init__()

        self.num_queries = args.num_queries
        self.backbone = build_unit_convnet_backbone(args)
        self.transformer = TRSTransformer(args)
        # self.transformer = TRSDeformableTransformer(args)
        encoder_hidden_dim = self.transformer.d_model_enc
        decoder_hidden_dim = self.transformer.d_model_dec

        self.query_embeds = nn.ModuleDict()
        for task_type in self.num_queries:
            task_dict = nn.ModuleDict()
            for dataset in self.num_queries[task_type]:
                task_dict[dataset] = nn.Embedding(
                    self.num_queries[task_type][dataset], decoder_hidden_dim
                )
            self.query_embeds[task_type] = task_dict

        # build the hsi linear projection
        hsi_lin_proj = nn.ModuleDict()
        for dataset_name in args.num_queries.get("hsi_cls", []):
            hsi_lin_proj[dataset_name] = nn.Conv2d(
                in_channels=config.heads["hsi_cls"][dataset_name]["num_channels"],
                out_channels=3, kernel_size=1,
            )
        self.hsi_lin_proj = hsi_lin_proj


    def forward(
        self,
        img_src: Tensor,
        task_type: str = "detection",
        dataset_name: str = "dior",
        task_idx: Optional[int] = None,
    ):
        all_img_src = [None]
        all_img_mask = [None]
        all_img_pos = [None]

        # get configs
        use_deformable_attention = self.use_deformable_attention[task_type]
        if not isinstance(use_deformable_attention, bool):
            use_deformable_attention = use_deformable_attention[dataset_name]
        multi_scale = self.multi_scale[task_type]
        if not isinstance(multi_scale, bool):
            multi_scale = multi_scale[dataset_name]
        two_stage = self.two_stage[task_type]
        if not isinstance(two_stage, bool):
            two_stage = two_stage[dataset_name]

        # Process image sources
        if img_src is not None:
            if not isinstance(img_src, NestedTensor):
                img_src = NestedTensor.from_tensor_list(img_src)

            # HSI Input Projection
            if task_type == 'hsi_cls':
                img_src.tensors = self.hsi_lin_proj[dataset_name](img_src.tensors)

            # Backbone: ResNet/VGG backbone + Conv Neck (channel mapper) + position embedding
            features, all_img_pos = self.backbone(img_src)

            # Collate the src, mask, and pos
            if not multi_scale:
                all_img_src, all_img_mask = features[-1].decompose()
                all_img_pos = all_img_pos[-1]
                if use_deformable_attention:
                    all_img_src, all_img_mask, all_img_pos = \
                        [all_img_src], [all_img_mask], [all_img_pos]
            else:
                # assert use_deformable_attention
                all_img_src = []
                all_img_mask = []
                for l, feat in enumerate(features):
                    src, mask = feat.decompose()
                    all_img_src.append(src)
                    all_img_mask.append(mask)

        # Get query_embed for Decoder
        if not two_stage:
            query_embed = self.query_embeds[task_type][dataset_name].weight
        else:
            assert use_deformable_attention
            raise NotImplementedError

        # Transformer
        hs, _ = self.transformer(
            img_src=all_img_src,
            img_mask=all_img_mask,
            img_pos=all_img_pos,
            query_embed=query_embed,
            task_type=task_type,
            dataset_name=dataset_name,
            task_idx=task_idx,
        )

        if hs is not None:
            assert hs.size(2) == self.num_queries[task_type][dataset_name]

        return {"hidden_states": hs}

