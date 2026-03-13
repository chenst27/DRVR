import torch
import torch.nn as nn
import torch_scatter
import spconv.pytorch as spconv


class SparseBasicBlock3D(spconv.SparseModule):
    def __init__(self, in_channel, out_channel, indice_key):
        super(SparseBasicBlock3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.in_conv = spconv.SubMConv3d(self.in_channel, self.out_channel, kernel_size=1, 
            stride=1, padding=0, indice_key=indice_key+"inconv", bias=False)
        self.in_bn = nn.BatchNorm1d(self.out_channel)

        self.conv1 = spconv.SubMConv3d(self.in_channel, self.out_channel, kernel_size=3, 
            stride=1, padding=1, indice_key=indice_key+"conv1", bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = spconv.SubMConv3d(self.out_channel, self.out_channel, kernel_size=3, 
            stride=1, padding=1, indice_key=indice_key+"conv2", bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channel)

    def forward(self, x):
        identity = self.in_conv(x)
        identity = identity.replace_feature(self.in_bn(identity.features))
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity.features))
        return out


class GeoEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, return_fea, num_classes, num_height, max_volume_space, min_volume_space, spatial_shape_list, height_maxpool_list):
        super(GeoEncoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_classes = num_classes
        self.num_height = num_height
        self.return_fea = return_fea
        self.max_volume_space = torch.tensor(max_volume_space)
        self.min_volume_space = torch.tensor(min_volume_space)
        self.spatial_shape_list = torch.tensor(spatial_shape_list)    # [[128,128,16], [64,64,8], [32,32,4]]
        self.height_maxpool_list = torch.tensor(height_maxpool_list)  # [4, 4, 4]
        assert len(spatial_shape_list) == len(height_maxpool_list)
        self.num_scales = len(spatial_shape_list)

        height_split_list = [int(self.spatial_shape_list[idx][-1] / self.height_maxpool_list[idx]) for idx in range(self.num_scales)]
        in_channel_bev = [int(out_channel * height) for height in height_split_list]
        out_channel_bev = [int(out_channel) for _ in range(self.num_scales)]

        self.pt_encoder = nn.ModuleList()
        self.spv_encoder = nn.ModuleList()
        self.spv_maxpool = nn.ModuleList()
        self.bev_encoder = nn.ModuleList()

        for idx in range(self.num_scales):
            self.pt_encoder.append(
                nn.Sequential(
                    nn.Linear(in_channel, out_channel),
                    nn.ReLU(True),
                    nn.Linear(out_channel, out_channel)
                )
            )
            self.spv_encoder.append(
                spconv.SparseSequential(
                    SparseBasicBlock3D(in_channel=out_channel, out_channel=out_channel, indice_key="spv_"+str(idx)+"_block1"),
                    SparseBasicBlock3D(in_channel=out_channel, out_channel=out_channel, indice_key="spv_"+str(idx)+"_block2")
                )
            )
            self.spv_maxpool.append(
                spconv.SparseMaxPool3d(
                    kernel_size=[1, 1, height_maxpool_list[idx]],
                    stride=[1, 1, height_maxpool_list[idx]],
                    padding=0
                )
            )
            self.bev_encoder.append(
                nn.Sequential(
                    nn.Linear(in_channel_bev[idx], out_channel_bev[idx]), 
                    nn.ReLU(), 
                    nn.Linear(out_channel_bev[idx], out_channel_bev[idx])
                )
            )

        self.up1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channel*3, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 1),
        )
        self.classifier = nn.Conv2d(out_channel, self.num_classes*self.num_height, 1)

        if self.return_fea:
            fea_dim = 32
            self.out_layer = nn.Sequential(
                nn.Conv2d(out_channel, fea_dim*self.num_height, 3, padding=1, bias=False),
                nn.BatchNorm2d(fea_dim*self.num_height),
                nn.ReLU(inplace=True),
                nn.Conv2d(fea_dim*self.num_height, fea_dim*self.num_height, 1)
            )


    def forward(self, data_dict):
        batch_size = data_dict['batch_size']
        xyz = data_dict['xyz']
        intensity = data_dict['intensity'].reshape(-1, 1)

        voxel_coors_list = data_dict['voxel_coors_list']
        coors_unq_list = data_dict['coors_unq_list']
        coors_inv_list = data_dict['coors_inv_list']

        bev_feats_list = []
        for idx in range(self.num_scales):
            voxel_coors = voxel_coors_list[idx].to(xyz.device)
            coors_unq = coors_unq_list[idx].to(xyz.device)
            coors_inv = coors_inv_list[idx].to(xyz.device)
            xyz_mean = torch_scatter.scatter_mean(xyz, coors_inv, dim=0)[coors_inv]
            pt_feats = torch.cat((xyz, intensity, xyz-xyz_mean, xyz-voxel_coors), dim=1)
            pt_feats = self.pt_encoder[idx](pt_feats)
            vox_feats = torch_scatter.scatter_mean(pt_feats, coors_inv, dim=0)
            spv_feats = spconv.SparseConvTensor(vox_feats, coors_unq, self.spatial_shape_list[idx], batch_size)
            spv_feats = self.spv_encoder[idx](spv_feats)
            bev_feats = self.spv_maxpool[idx](spv_feats).dense().permute(0,2,3,4,1).flatten(start_dim=3).contiguous()
            bev_feats = self.bev_encoder[idx](bev_feats).permute(0,3,1,2).contiguous()
            bev_feats_list.append(bev_feats)

        feats_up1 = self.up1(bev_feats_list[0])
        feats_up2 = self.up2(bev_feats_list[1])
        feats_up3 = self.up3(bev_feats_list[2])
        feats_all = torch.cat([feats_up1, feats_up2, feats_up3], dim=1)

        out_feats = self.conv_out(feats_all)
        pred = self.classifier(out_feats)
        B, C, H, W = pred.shape
        pred = pred.permute(0,2,3,1).reshape(B, H, W, self.num_height, -1).contiguous()
        pred = pred.permute(0,4,1,2,3).contiguous()

        if self.return_fea:
            geo_feats = self.out_layer(out_feats)
            return pred, geo_feats
        else:
            return pred