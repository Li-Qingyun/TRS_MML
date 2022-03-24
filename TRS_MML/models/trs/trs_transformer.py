import torch
from typing import Optional
from torch import nn, Tensor
from mmf.models.unit.transformer import (Transformer,
                                         TransformerEncoder,
                                         TransformerDecoder,
                                         TransformerDecoderLayer,
                                         TransformerEncoderLayer)
from mmdet.models.utils.transformer import (DeformableDetrTransformer,
                                            DetrTransformerEncoder,
                                            DeformableDetrTransformerDecoder)
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead
from mmdet.models.detectors import DeformableDETR, BaseDetector, SingleStageDetector
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)


class TRSTransformer(nn.Module):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.d_model_enc = args.encoder_hidden_dim
        self.d_model_dec = args.decoder_hidden_dim
        self.dropout = args.dropout
        self.nhead = args.nheads
        self.dim_feedforward = args.dim_feedforward
        self.num_encoder_layers = args.enc_layers
        self.num_decoder_layers = args.dec_layers
        self.normalize_before = args.pre_norm
        self.return_intermediate_dec = True
        self.pass_pos_and_query = args.pass_pos_and_query
        self.share_decoders = args.share_decoders
        self.activation = "relu"

        self.pass_pos_and_query = self.pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(
            self.d_model_enc,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            self.activation,
            self.normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.d_model_enc) if self.normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm
        )

        if self.d_model_dec != self.d_model_enc:
            self.enc2dec_proj = nn.Linear(self.d_model_enc, self.d_model_dec)
            self.pos_embed_proj = nn.Linear(self.d_model_enc, self.d_model_dec)
        else:
            self.enc2dec_proj = nn.Identity()
            self.pos_embed_proj = nn.Identity()

        if self.share_decoders:
            decoder_layer = TransformerDecoderLayer(
                self.d_model_dec,
                self.nhead,
                self.dim_feedforward,
                self.dropout,
                self.activation,
                self.normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.d_model_dec)
            # self.decoder = TransformerDecoder(
            #     decoder_layer,
            #     self.num_decoder_layers,
            #     decoder_norm,
            #     return_intermediate=self.return_intermediate_dec,
            # )

        self._reset_parameters()

        # TODO: Son

        self.encoder_use_deformable_attention = args.encoder_use_deformable_attention
        self.decoder_use_deformable_attention = args.decoder_use_deformable_attention
        self.multi_scale = args.multi_scale
        self.two_stage = args.two_stage
        self.two_stage_num_proposals = args.two_stage_num_proposals

        num_queries = self.args.num_queries

        # Get deformable builder
        deformable = TRSDeformableBuilder(args)

        # Build the transformer modules
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for task in num_queries:
            encoder_dict = nn.ModuleDict()
            decoder_dict = nn.ModuleDict()
            for dataset in num_queries[task]:

                if self.encoder_use_deformable_attention[task]:
                    if not hasattr(self, 'deformable_encoder'):
                        self.deformable_encoder = deformable.build_encoder()
                    encoder_dict[dataset] = self.deformable_encoder
                else:
                    if not hasattr(self, 'encoder'):
                        raise NotImplementedError  # TODO: Modified encoder builder
                    encoder_dict[dataset] = self.encoder

                if self.decoder_use_deformable_attention[task]:
                    if self.share_decoders:
                        if not hasattr(self, 'deformable_decoder'):
                            self.deformable_decoder = deformable.build_decoder()
                        decoder_dict[dataset] = self.deformable_decoder
                    else:
                        decoder_dict[dataset] = deformable.build_decoder()
                else:
                    if self.share_decoders:
                        decoder_dict[dataset] = self.decoder
                    else:
                        decoder_dict[dataset] = self.build_decoder(
                            d_model_dec=self.d_model_dec,
                            nhead=self.nhead,
                            dim_feedforward=self.dim_feedforward,
                            dropout=self.dropout,
                            activation=self.activation,
                            normalize_before=self.normalize_before,
                            num_decoder_layers=self.num_decoder_layers,
                            return_intermediate_dec=self.return_intermediate_dec,
                        )
                self.encoders[task] = encoder_dict
                self.decoders[task] = decoder_dict

        # Init the task embedding
        MAX_TASK_NUM = 256
        if args.use_task_embedding_in_img_encoder:
            self.task_embeddings_enc = nn.Embedding(MAX_TASK_NUM, self.d_model_enc)
        # when adding the task embedding to the beginning of the decoder, we'll strip
        # it from the hidden state outputs to make it compatible with previous models
        self.mem_out_begin_idx = 1 if args.use_task_embedding_in_img_encoder else 0

        # Build the level embedding
        num_feature_levels = args.num_feature_levels
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, args.decoder_hidden_dim))
        nn.init.normal_(self.level_embed)

        #
        self.reference_points = nn.Linear(args.decoder_hidden_dim, 2)  # TODO: why

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_decoder(
        self,
        d_model_dec=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        decoder_layer = TransformerDecoderLayer(
            d_model_dec, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model_dec)
        return TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

    def forward(
        self,
        all_img_src: Optional[Tensor] = None,
        all_img_mask: Optional[Tensor] = None,
        all_img_pos: Optional[Tensor] = None,
        query_embed: Optional[Tensor] = None,
        reg_branches: Optional[nn.ModuleList] = None,
        cls_branches: Optional[nn.ModuleList] = None,
        task_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        task_idx: Optional[int] = None,
        encoder_use_deformable_attention = False,
        decoder_use_deformable_attention = False,
        multi_scale = False,
        two_stage = False,
        with_box_refine = False,
    ):

        # get configs    # TODO:有没有必要每个数据集设置接口不一样
        encoder_use_deformable_attention = self.encoder_use_deformable_attention[task_type]
        if not isinstance(encoder_use_deformable_attention, bool):
            encoder_use_deformable_attention = encoder_use_deformable_attention[dataset_name]
        decoder_use_deformable_attention = self.decoder_use_deformable_attention[task_type]
        if not isinstance(decoder_use_deformable_attention, bool):
            decoder_use_deformable_attention = decoder_use_deformable_attention[dataset_name]
        multi_scale = self.multi_scale[task_type]
        if not isinstance(multi_scale, bool):
            multi_scale = multi_scale[dataset_name]
        two_stage = self.two_stage[task_type]
        if not isinstance(two_stage, bool):
            two_stage = two_stage[dataset_name]

        # get encoder and decoder
        encoder = self.encoders[task_type][dataset_name]
        decoder = self.decoders[task_type][dataset_name]

        # TODO: implement multimodal input
        memories = []
        pos_embeds = []
        masks = []

        all_src_flatten = []
        all_mask_flatten = []
        all_pos_embed_flatten = []
        spatial_shapes = []
        reference_points = None
        level_start_index = None
        valid_ratios = None

        # Prepare input for encoder
        for lvl, (src, mask, pos_embed) in enumerate(zip(
                all_img_src, all_img_mask, all_img_pos
        )):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # flatten NxCxHxW to HWxNxC
            src = src.flatten(2).permute(2, 0, 1)  #.transpose(1, 2) --> (0, 2, 1)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  #.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # 应该不用改
            all_pos_embed_flatten.append(lvl_pos_embed)
            all_src_flatten.append(src)
            all_mask_flatten.append(mask)
        all_src_flatten = torch.cat(all_src_flatten, 0)  #1)
        all_mask_flatten = torch.cat(all_mask_flatten, 1)
        all_pos_embed_flatten = torch.cat(all_pos_embed_flatten, 0)  #1)
        if encoder_use_deformable_attention or decoder_use_deformable_attention:
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=all_src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in all_img_mask], 1)
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=all_src_flatten.device)

        if self.pass_pos_and_query:
            tgt = torch.zeros_like(query_embed)
        else:
            all_src_flatten, tgt, query_embed, all_img_pos = (
                all_src_flatten + 0.1 * all_img_pos,
                query_embed,
                None,
                None,
            )

        all_src_flatten, all_mask_flatten, all_pos_embed_flatten, level_start_index = \
            self._prefix_task_embedding_to_encoder_inputs(  #  TODO: deformable的情况下，现在还不能做 task embedding
            all_src_flatten, all_mask_flatten, all_pos_embed_flatten, level_start_index, task_idx
        )  # Prefix task embedding

        # Encoder
        if not encoder_use_deformable_attention:  # TODO 检查一下维度再输入
            assert isinstance(encoder, TransformerEncoder)
            memory = encoder(
                src=all_src_flatten,
                src_key_padding_mask=all_mask_flatten,
                pos=all_pos_embed_flatten,
            )
        else:
            assert isinstance(encoder, DetrTransformerEncoder)  # deformable
            memory = encoder(
                query=all_src_flatten,
                key=None,
                value=None,
                query_pos=all_pos_embed_flatten,
                query_key_padding_mask=all_mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        # Prepare input for decoder
        # bs, _, c = memory.shape
        _, bs, c = memory.shape

        if two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, all_mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(1).expand(-1, bs, -1)
            tgt = tgt.unsqueeze(1).expand(-1, bs, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        if self.mem_out_begin_idx != 0:  # TODO:???
            all_img_src = all_img_src[self.mem_out_begin_idx :]
            all_img_pos = all_img_pos[self.mem_out_begin_idx :]
            all_img_mask = all_img_mask[:, self.mem_out_begin_idx :]
            memory = memory[self.mem_out_begin_idx :]

        if self.args.residual_in_encoder:    # TODO:???
            memory = all_img_src + memory

        memory = self.enc2dec_proj(memory)
        all_img_pos = self.pos_embed_proj(all_img_pos)

        memories.append(memory)
        pos_embeds.append(all_pos_embed_flatten)
        masks.append(all_mask_flatten)

        decoder = self.decoders[task_type][dataset_name]

        memories = torch.cat(memories)
        masks = torch.cat(masks, dim=-1)
        pos_embeds = torch.cat(pos_embeds)

        if not decoder_use_deformable_attention:
            # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # TODO: implement multimodal concat
            assert isinstance(decoder, TransformerDecoder)
            hs = decoder(
                tgt=tgt,
                memory=memories,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=masks,
                pos=pos_embeds,
                query_pos=query_embed,
            )
        else:
            assert isinstance(decoder, DeformableDetrTransformerDecoder)
            hs, inter_references = decoder(
                query=tgt,
                key=None,
                value=memories,
                query_pos=query_embed,
                key_padding_mask=all_mask_flatten,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reg_branches=reg_branches)

        # hs is num_layer x batch_size x seq_length x hidden_dim

        return hs.transpose(1, 2), memories.permute(1, 2, 0)

    def _prefix_task_embedding_to_encoder_inputs(
        self, img_src, img_mask, img_pos, lvl_start_index, task_idx
    ):
        if not self.args.use_task_embedding_in_img_encoder:
            return img_src, img_mask, img_pos, lvl_start_index

        bs = img_src.size(1)
        task_embed = self.task_embeddings_enc.weight[task_idx]
        task_embed = task_embed.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1)
        img_src = torch.cat([task_embed, img_src], dim=0)

        # 0 for non-padding in img_mask
        img_mask_pad = torch.zeros_like(img_mask[:, :1])
        img_mask = torch.cat([img_mask_pad, img_mask], dim=1)
        img_pos_pad = torch.zeros_like(img_pos[:1])
        img_pos = torch.cat([img_pos_pad, img_pos], dim=0)

        for i, ind in enumerate(lvl_start_index):
            lvl_start_index[i] = ind + 1

        return img_src, img_mask, img_pos, lvl_start_index

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class TRSDeformableBuilder:

    def __init__(self, args):
        super().__init__()

        self.args = args  # TODO
        self.d_model_enc = args.encoder_hidden_dim
        self.d_model_dec = args.decoder_hidden_dim
        self.dropout = args.dropout
        self.nhead = args.nheads
        self.dim_feedforward = args.dim_feedforward
        self.num_encoder_layers = args.enc_layers
        self.num_decoder_layers = args.dec_layers
        self.normalize_before = args.pre_norm  # TODO
        self.return_intermediate_dec = True  # TODO
        self.pass_pos_and_query = args.pass_pos_and_query  # TODO
        self.share_decoders = args.share_decoders  # TODO
        self.activation = "relu"  # TODO
        self.num_feature_levels = args.num_feature_levels

        self.configdict = self.build_configdict(**args)

    def build_encoder(self):
        return build_transformer_layer_sequence(self.configdict.encoder)

    def build_decoder(self):
        return build_transformer_layer_sequence(self.configdict.decoder)

    def build_transformer(self):
        return DeformableDetrTransformer(**self.configdict)

    def build_configdict(self, **args):
        from mmcv.utils import ConfigDict
        config = ConfigDict(
            dict(
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=self.num_encoder_layers,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=self.d_model_enc,
                            num_levels=self.num_feature_levels,
                        ),
                        feedforward_channels=self.dim_feedforward,
                        ffn_dropout=self.dropout,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder',
                    num_layers=self.num_decoder_layers,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=self.d_model_dec,
                                num_heads=self.nhead,
                                dropout=self.dropout),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=self.d_model_enc,
                                num_levels=self.num_feature_levels,)
                        ],
                        feedforward_channels=self.dim_feedforward,
                        ffn_dropout=self.dropout,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')))))
        return config