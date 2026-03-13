import torch
import torch.nn as nn 
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter

from models.CENet import ResNet_34, convert_relu_to_softplus, load_partial_weights
from models.deformable_transformer import DeformableTransformer, DeformableTransformerLayer
from models.vox_fea_extractor import SparseVFE
from models.geometric_encoder import GeoEncoder


class DRVR(nn.Module):
    def __init__(self, config):
        super(DRVR, self).__init__()
        self.config = config
        self.sparse_shape = self.config['model_params']['query_shape']
        self.bev_h = self.sparse_shape[0]
        self.bev_w = self.sparse_shape[1]
        self.bev_z = self.sparse_shape[2]
        self.embed_dims = self.config['model_params']['cross_attn_embed_dims']
        self.num_classes = self.config['model_params']['num_classes']
        self.pretrained_model = self.config['model_params']['pretrained_model']

        # CENet
        range_encoder = ResNet_34()
        convert_relu_to_softplus(range_encoder, nn.Hardswish())
        self.range_encoder = load_partial_weights(
            model=range_encoder, 
            ckpt_path=self.pretrained_model
        )

        self.geo_encoder = GeoEncoder(
            in_channel=10, 
            out_channel=self.embed_dims, 
            num_classes=2, 
            return_fea=True,
            num_height=self.bev_z,
            max_volume_space=self.config['dataset_params']['max_volume_space'], 
            min_volume_space=self.config['dataset_params']['min_volume_space'], 
            spatial_shape_list=self.config['model_params']['spatial_shape_list'], 
            height_maxpool_list=self.config['model_params']['height_maxpool_list']
        )

        self.proj_layer = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(), 
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, self.embed_dims)
        )
        
        cross_deform_attn_layer_1_2 = DeformableTransformerLayer(
            d_model=self.embed_dims, 
            d_ffn=self.config['model_params']['cross_attn_ffn_dims'],
            dropout=0.1, 
            activation="relu",
            n_levels=self.config['model_params']['cross_attn_n_levels_1_2'], 
            n_heads=self.config['model_params']['cross_attn_n_heads'], 
            n_points=self.config['model_params']['cross_attn_n_points']
        )
        self.cross_deform_attn_1_2 = DeformableTransformer(
            transformer_layer=cross_deform_attn_layer_1_2, 
            num_transformer_layers=self.config['model_params']['cross_attn_n_layers_1_2'], 
            num_feature_levels=self.config['model_params']['cross_attn_n_levels_1_2'], 
            embed_dims=self.embed_dims
        )

        self.fusion_layer = spconv.SparseSequential(
            spconv.SubMConv3d(self.embed_dims*2, self.embed_dims, indice_key="fusion_layer_1", kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(self.embed_dims, self.embed_dims, indice_key="fusion_layer_2", kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.fine_sparse_decoder_1_2 = SparseVFE(in_channel=self.embed_dims, out_channel=self.embed_dims, indice_key='fine_sparse_decoder_1_2')
        self.fine_classifier_1_1 = spconv.SparseSequential(
            spconv.SubMConv3d(self.embed_dims, self.embed_dims, indice_key="fine_classifier_1_1_1", kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(self.embed_dims, self.num_classes, indice_key="fine_classifier_1_1_2", kernel_size=3, stride=1, padding=1, bias=True)
        )


    def forward(self, data_dict):
        rv_feats = self.range_encoder(x=data_dict['proj'])
        pad = nn.ZeroPad2d(padding=(int(rv_feats.shape[-1] / 2), int(rv_feats.shape[-1] / 2), 0, 0))
        rv_feats_new = [pad(rv_feats)]

        sc_logits_1_2, bev_geo_feats = self.geo_encoder(data_dict)
        B, _, H, W, D = sc_logits_1_2.shape
        sc_logits_1_2_tmp = sc_logits_1_2.permute(0, 2, 3, 4, 1).flatten(0, 3)
        sc_logits_1_2_tmp = sc_logits_1_2_tmp.argmax(dim=1)
        nonempty_mask_1_2 = (sc_logits_1_2_tmp != 0)

        B, _, H, W = bev_geo_feats.shape
        vox_geo_feats = bev_geo_feats.permute(0,2,3,1).reshape(B, H, W, self.bev_z, -1).flatten(0,3)
        vox_queries_1_2 = vox_geo_feats[nonempty_mask_1_2]
        vox_queries_1_2 = self.proj_layer(vox_queries_1_2)
        vox_queries_1_2 = vox_queries_1_2.unsqueeze(0)

        # forward projection
        vox_feats_1_2_all = torch.zeros(((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims), device=rv_feats_new[0].device)
        ref_points = data_dict['proj_vox_centers_1_2'][:, nonempty_mask_1_2].cuda()  # (bs, n_query, n_level, 2)

        vox_feats_1_2 = self.cross_deform_attn_1_2(
            query=vox_queries_1_2, 
            feats=rv_feats_new, 
            reference_points=ref_points.float()
            )   # (bs, 128*128*16, c)
        vox_feats_1_2_all[nonempty_mask_1_2] = vox_feats_1_2[0]
        vox_feats_1_2_all = vox_feats_1_2_all.unsqueeze(0)
        vox_feats_1_2_all = vox_feats_1_2_all.reshape(1, self.bev_h, self.bev_w, self.bev_z, -1).permute(0, 4, 1, 2, 3)
        coors_unq_1_2, sparse_feats_1_2 = extract_nonzero_features(vox_feats_1_2_all)
        # sparse_feats_1_2 = spconv.SparseConvTensor(sparse_feats_1_2, coors_unq_1_2, [128,128,16], 1)

        # backward projection
        proj_xy = data_dict['proj_xy'][0].long()
        range_feats = rv_feats_new[0][0, :, proj_xy[:, 1], proj_xy[:, 0]]
        range_feats = range_feats.permute(1, 0)
        point_coors_unq = data_dict['point_coors_unq']
        point_coors_inv = data_dict['point_coors_inv']
        point_feats = torch_scatter.scatter_mean(range_feats, point_coors_inv.cuda(), dim=0)
        point_feats = spconv.SparseConvTensor(point_feats, point_coors_unq, [128,128,16], 1)
        point_feats = point_feats.dense().permute(0,2,3,4,1).flatten(0,3)
        sparse_point_feats = point_feats[nonempty_mask_1_2]

        geo_feats = vox_queries_1_2[0] + sparse_point_feats
        sparse_geo_feats = spconv.SparseConvTensor(geo_feats, coors_unq_1_2, [128,128,16], 1)
        sparse_geo_feats = self.fine_sparse_decoder_1_2(sparse_geo_feats)

        fused_feats = torch.cat((sparse_geo_feats.features, sparse_feats_1_2), dim=-1)
        sparse_fused_feats = spconv.SparseConvTensor(fused_feats, coors_unq_1_2, [128,128,16], 1)
        sparse_fused_feats = self.fusion_layer(sparse_fused_feats)

        dense_fine_feats_1_2 = sparse_fused_feats.dense()
        dense_fine_feats_1_2 = dense_fine_feats_1_2.flatten(2).permute(0, 2, 1)
        dense_fine_feats_1_1 = dense_fine_feats_1_2[:, data_dict['f2cmapping_1_2'][0], :]
        dense_fine_feats_1_1 = dense_fine_feats_1_1.reshape(1, 256, 256, 32, -1).permute(0, 4, 1, 2, 3)
        coors_unq_1_1, sparse_feats_1_1 = extract_nonzero_features(dense_fine_feats_1_1)
        sparse_feats_tensor = spconv.SparseConvTensor(sparse_feats_1_1, coors_unq_1_1, [256,256,32], 1)
        fine_logits_1_1 = self.fine_classifier_1_1(sparse_feats_tensor)
        ssc_logits_1_1 = fine_logits_1_1.dense()

        # get mask of predictions
        fine_logits = fine_logits_1_1.dense().clone()
        fine_logits = fine_logits.flatten(2).permute(0, 2, 1)
        nonzero_index = torch.sum(torch.abs(fine_logits[0]), dim=1) != 0

        preds = {}
        preds['ssc_logits_1_1'] = ssc_logits_1_1        # (bs, n_class, 256, 256, 32)
        preds['sc_logits_1_2'] = sc_logits_1_2          # (bs, 2, 128, 128, 16)
        preds['num_nonempty_1_1'] = nonzero_index.sum()
        preds['num_nonempty_1_2'] = nonempty_mask_1_2.sum()
        preds['pred_mask_1_1'] = nonzero_index          # (256*256*32, )
        return preds


def extract_nonzero_features(x):
    device = x.device
    nonzero_index = torch.sum(torch.abs(x), dim=1).nonzero()
    coords = nonzero_index.type(torch.int32).to(device)
    channels = int(x.shape[1])
    features = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    features = features[torch.sum(torch.abs(features), dim=1).nonzero(), :]
    features = features.squeeze(1).to(device)
    coords, _, _ = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
    return coords, features 