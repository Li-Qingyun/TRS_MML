from typing import Optional
from torch import Tensor, nn
from mmf.models.unit.misc import NestedTensor
from TRS_MML.models.trs.trs_transformer import TRSTransformer
from mmf.models.unit.backbone import build_unit_convnet_backbone


class TRSBaseModel(nn.Module):
    def __init__(self, config):
        args = config.base_args
        super().__init__()

        self.num_queries = args.num_queries
        self.backbone = build_unit_convnet_backbone(args)
        self.transformer = TRSTransformer(args)
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

        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, encoder_hidden_dim, kernel_size=1
        )

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
        img_mask = None
        img_pos = [None]
        if img_src is not None:
            if not isinstance(img_src, NestedTensor):
                img_src = NestedTensor.from_tensor_list(img_src)

            if task_type == 'hsi_cls':
                img_src.tensors = self.hsi_lin_proj[dataset_name](img_src.tensors)

            features, img_pos = self.backbone(img_src)

            img_src, img_mask = features[-1].decompose()
            img_src = self.input_proj(img_src)

        query_embed = self.query_embeds[task_type][dataset_name]
        hs, _ = self.transformer(
            img_src=img_src,
            img_mask=img_mask,
            img_pos=img_pos[-1],
            query_embed=query_embed.weight,
            task_type=task_type,
            dataset_name=dataset_name,
            task_idx=task_idx,
        )

        if hs is not None:
            assert hs.size(2) == self.num_queries[task_type][dataset_name]

        return {"hidden_states": hs}

