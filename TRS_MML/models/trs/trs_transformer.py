import torch
from typing import Optional
from torch import nn, Tensor
from mmf.models.unit.transformer import Transformer, TransformerDecoder, TransformerDecoderLayer


class TRSTransformer(Transformer):
    def __init__(self, args):
        super().__init__(args=args)

        num_queries = self.args.num_queries
        self.decoders = nn.ModuleDict()
        for task in num_queries:
            task_dict = nn.ModuleDict()
            for dataset in num_queries[task]:
                if self.share_decoders:
                    task_dict[dataset] = self.decoder
                else:
                    task_dict[dataset] = self.build_decoder_layer(
                        d_model_dec=self.d_model_dec,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        activation=self.activation,
                        normalize_before=self.normalize_before,
                        num_decoder_layers=self.num_decoder_layers,
                        return_intermediate_dec=self.return_intermediate_dec,
                    )
            self.decoders[task] = task_dict
            # A separate decoder for VQA

        MAX_TASK_NUM = 256
        if args.use_task_embedding_in_img_encoder:
            self.task_embeddings_enc = nn.Embedding(MAX_TASK_NUM, self.d_model_enc)
        # when adding the task embedding to the beginning of the decoder, we'll strip
        # it from the hidden state outputs to make it compatible with previous models
        self.mem_out_begin_idx = 1 if args.use_task_embedding_in_img_encoder else 0

    def build_decoder_layer(
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
        img_src: Optional[Tensor] = None,
        img_mask: Optional[Tensor] = None,
        img_pos: Optional[Tensor] = None,
        query_embed: Optional[Tensor] = None,
        task_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        task_idx: Optional[int] = None,
    ):
        # flatten NxCxHxW to HWxNxC
        memories = []
        pos_embeds = []
        masks = []

        if img_src is not None:
            bs, c, h, w = img_src.shape
            img_src = img_src.flatten(2).permute(2, 0, 1)
            img_pos = img_pos.flatten(2).permute(2, 0, 1)
            img_mask = img_mask.flatten(1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                img_src, tgt, query_embed, img_pos = (
                    img_src + 0.1 * img_pos,
                    query_embed,
                    None,
                    None,
                )
            img_src, img_mask, img_pos = self._prefix_task_embedding_to_encoder_inputs(
                img_src, img_mask, img_pos, task_idx
            )
            memory = self.encoder(img_src, src_key_padding_mask=img_mask, pos=img_pos)

            if self.mem_out_begin_idx != 0:
                img_src = img_src[self.mem_out_begin_idx :]
                img_pos = img_pos[self.mem_out_begin_idx :]
                img_mask = img_mask[:, self.mem_out_begin_idx :]
                memory = memory[self.mem_out_begin_idx :]

            if self.args.residual_in_encoder:
                memory = img_src + memory

            memory = self.enc2dec_proj(memory)
            img_pos = self.pos_embed_proj(img_pos)
            memories.append(memory)
            pos_embeds.append(img_pos)
            masks.append(img_mask)

        decoder = self.decoders[task_type][dataset_name]

        memories = torch.cat(memories)
        masks = torch.cat(masks, dim=-1)
        pos_embeds = torch.cat(pos_embeds)

        hs = decoder(
            tgt,
            memories,
            memory_key_padding_mask=masks,
            pos=pos_embeds,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        # hs is num_layer x batch_size x seq_length x hidden_dim

        return hs, memories.permute(1, 2, 0)

    def _prefix_task_embedding_to_encoder_inputs(
        self, img_src, img_mask, img_pos, task_idx
    ):
        if not self.args.use_task_embedding_in_img_encoder:
            return img_src, img_mask, img_pos

        bs = img_src.size(1)
        task_embed = self.task_embeddings_enc.weight[task_idx]
        task_embed = task_embed.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1)
        img_src = torch.cat([task_embed, img_src], dim=0)

        # 0 for non-padding in img_mask
        img_mask_pad = torch.zeros_like(img_mask[:, :1])
        img_mask = torch.cat([img_mask_pad, img_mask], dim=1)
        img_pos_pad = torch.zeros_like(img_pos[:1])
        img_pos = torch.cat([img_pos_pad, img_pos], dim=0)

        return img_src, img_mask, img_pos