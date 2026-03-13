import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

# from models.ops.modules import MSDeformAttn
from mmcv.ops import MultiScaleDeformableAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class DeformableTransformerLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=d_model, 
            num_heads=n_heads,
            num_levels=n_levels,
            num_points=n_points,
            batch_first=True
            )

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        query2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm2(query)
        return query

    def forward(self, query, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # cross attention
        # query2 = self.cross_attn(self.with_pos_embed(query, query_pos),
        #                         reference_points,
        #                         src, src_spatial_shapes, level_start_index, src_padding_mask)
        query2 = self.cross_attn(
            query=query,
            key=None,
            value=src,
            identity=None,
            query_pos=query_pos,
            key_padding_mask=src_padding_mask,
            reference_points=reference_points,
            spatial_shapes=src_spatial_shapes,
            level_start_index=level_start_index,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)
        
        return query


class DeformableTransformer(nn.Module):
    def __init__(self, transformer_layer, num_transformer_layers, num_feature_levels, embed_dims):
        super().__init__()
        self.layers = _get_clones(transformer_layer, num_transformer_layers)
        self.num_layers = num_transformer_layers
        self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))
        normal_(self.level_embeds)

    def forward(self, query, feats, reference_points, query_pos=None):
        output = query

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(0, 2, 1)
            feat = feat + self.level_embeds[None, lvl:lvl + 1, :].to(feat.dtype)
            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # cross attention
        for lid, layer in enumerate(self.layers):
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                src=feat_flatten,
                src_spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )

        return output
        