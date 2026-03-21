import os
import numpy as np
import torch
from torch.utils import data
import json
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


class NuscenesOcc3D(data.Dataset):
    def __init__(self, config, dataset_type, flip_aug=False):
        self.config = config
        self.flip_aug = flip_aug

        self.data_path = self.config['dataset_params']['data_path']
        self.label_path = self.config['dataset_params']['label_path']
        self.anno_path = self.config['dataset_params']['anno_path']
        self.mask_camera = self.config['dataset_params']['mask_camera']
        self.grid_shape_list = np.asarray(self.config['dataset_params']['grid_shape_list'])
        self.max_volume_space = np.asarray(self.config['dataset_params']['max_volume_space'])
        self.min_volume_space = np.asarray(self.config['dataset_params']['min_volume_space'])

        self.proj_h = self.config['dataset_params']['proj_img_size'][0]
        self.proj_w = self.config['dataset_params']['proj_img_size'][1]
        self.range_img_means = torch.tensor(self.config['dataset_params']["range_img_means"], dtype=torch.float)
        self.range_img_stds = torch.tensor(self.config['dataset_params']["range_img_stds"], dtype=torch.float) 

        self.fov_up = self.config['dataset_params']['fov_up'] / 180.0 * np.pi
        self.fov_down = self.config['dataset_params']['fov_down'] / 180.0 * np.pi
        self.fov_v = abs(self.fov_up) + abs(self.fov_down)

        self.class_names = [ 'free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',  
                             'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',  
                             'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'others']

        with open(self.anno_path, 'r') as anno_file:
            self.annotations = json.load(anno_file)
        if dataset_type == "train":
            self.scene_list = self.annotations['train_split']
        else:
            self.scene_list = self.annotations['val_split']

        self.data_info = []
        for scene_index in self.scene_list:
            for sample_token in self.annotations['scene_infos'][scene_index].keys():
                sample_info = {
                    'sample_token': sample_token,
                    'scene_index': scene_index
                }
                self.data_info.append(sample_info)

        self.nusc = NuScenes(
            version='v1.0-trainval', dataroot=self.data_path, verbose=False)

        self.n_levels_1_2 = self.config['model_params']['cross_attn_n_levels_1_2']
        self.voxel_centers_1_2 = self.get_vox_centers(grid_shape=[100, 100, 8])

        # get mapping
        voxel_centers_1_1 = self.get_vox_centers(grid_shape=[200, 200, 16])
        self.f2cmapping_1_2 = self.computeFine2CoarseMapping(
            fine_voxel_centers=voxel_centers_1_1, 
            coarse_grid_shape=[100, 100, 8], coarse_grid_size=0.8
        )

        self.grid_shape = np.array([100, 100, 8])


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_info)


    def __getitem__(self, index):
        
        scene_index = self.data_info[index]['scene_index']
        sample_token = self.data_info[index]['sample_token']

        # get occupancy label
        sample_info = self.annotations['scene_infos'][scene_index][sample_token]
        gt = np.load(os.path.join(self.label_path, sample_info['gt_path']))
        voxel_label_1_1 = gt['semantics']
        if self.mask_camera:
            voxel_label_1_1[np.isclose(gt['mask_camera'], 0)] = 255

        # get pointclouds
        sample_data = self.nusc.get('sample', sample_token)
        lidar_data = self.nusc.get('sample_data', sample_data['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.data_path, lidar_data['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
        xyz = raw_data[:, :3]
        intensity = raw_data[:, 3]
        xyz_proj = xyz.copy()
        intensity_proj = intensity.copy()

        lidar2ego = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        xyz_ego = self.lidar_to_ego_transform(xyz, lidar2ego)
        mask_x = np.logical_and(xyz_ego[:, 0] > self.min_volume_space[0], xyz_ego[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz_ego[:, 1] > self.min_volume_space[1], xyz_ego[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz_ego[:, 2] > self.min_volume_space[2], xyz_ego[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
        
        xyz = xyz[mask]
        xyz_ego = xyz_ego[mask]
        intensity = intensity[mask]
        
        if self.flip_aug and np.random.rand() > 0.5:
            xyz[:, 1] = -xyz[:, 1]
            xyz_proj[:, 1] = -xyz_proj[:, 1]
            xyz_ego[:, 1] = -xyz_ego[:, 1]
            voxel_label_1_1_tmp = voxel_label_1_1.copy()
            for y in range(voxel_label_1_1_tmp.shape[1]):
                voxel_label_1_1_tmp[:, y, :] = voxel_label_1_1[:, voxel_label_1_1_tmp.shape[1]-1-y, :]
            voxel_label_1_1 = voxel_label_1_1_tmp

        if self.flip_aug and np.random.rand() > 0.5:
            xyz[:, 0] = -xyz[:, 0]
            xyz_proj[:, 0] = -xyz_proj[:, 0]
            xyz_ego[:, 0] = -xyz_ego[:, 0]
            voxel_label_1_1_tmp = voxel_label_1_1.copy()
            for x in range(voxel_label_1_1_tmp.shape[0]):
                voxel_label_1_1_tmp[x, :, :] = voxel_label_1_1[voxel_label_1_1_tmp.shape[0]-1-x, :, :]
            voxel_label_1_1 = voxel_label_1_1_tmp
            
        voxel_label_1_1 = torch.from_numpy(voxel_label_1_1).float()
        mask_0 = (voxel_label_1_1 == 0)
        mask_17 = (voxel_label_1_1 == 17)
        voxel_label_1_1[mask_0] = 17
        voxel_label_1_1[mask_17] = 0

        voxel_label_1_2 = voxel_label_1_1.clone().reshape(100, 2, 100, 2, 8, 2).permute(0,2,4,1,3,5).reshape(100, 100, 8, 8)
        empty_mask_1_2 = voxel_label_1_2.sum(-1) == 0
        voxel_label_1_2 = voxel_label_1_2.to(torch.int64)
        occ_space_1_2 = voxel_label_1_2[~empty_mask_1_2]
        occ_space_1_2[occ_space_1_2==0] = -torch.arange(len(occ_space_1_2[occ_space_1_2==0])).to(occ_space_1_2.device) - 1
        voxel_label_1_2[~empty_mask_1_2] = occ_space_1_2
        voxel_label_1_2 = torch.mode(voxel_label_1_2, dim=-1)[0]
        voxel_label_1_2[voxel_label_1_2<0] = 255
        voxel_label_1_2 = voxel_label_1_2.long()

        xyz_raw = xyz_ego.copy()
        intensity_raw = intensity.copy()

        grid_index_list = []
        voxel_coors_list = []
        for idx in range(len(self.grid_shape_list)):
            intervals = (self.max_volume_space - self.min_volume_space) / (self.grid_shape_list[idx])
            if (intervals == 0).any(): print("Zero interval!")
            grid_index_ = (np.floor((xyz_ego.copy() - self.min_volume_space) / intervals)).astype(np.int32)
            voxel_coors = (grid_index_.astype(np.float32) + 0.5) * intervals + self.min_volume_space
            grid_index_list.append(grid_index_)
            voxel_coors_list.append(voxel_coors)
        grid_index = np.stack(grid_index_list, axis=0)
        voxel_coors = np.stack(voxel_coors_list, axis=0)

        intervals = (self.max_volume_space - self.min_volume_space) / (self.grid_shape)
        if (intervals == 0).any(): print("Zero interval!")
        point_index = (np.floor((xyz_ego.copy() - self.min_volume_space) / intervals)).astype(np.int32)

        # perform range_projection
        proj_x_sam, proj_y_sam = self.range_projection(xyz)

        proj_x_sam *= self.proj_w
        proj_y_sam *= self.proj_h
        proj_x_sam = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x_sam)), 0).astype(np.int32)
        proj_y_sam = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y_sam)), 0).astype(np.int32)
        
        proj_xy = np.concatenate((proj_x_sam.reshape(-1,1), proj_y_sam.reshape(-1,1)), axis=1)
        proj_xy = torch.from_numpy(proj_xy)
        
        # perform range_projection
        proj_x, proj_y = self.range_projection(xyz_proj)

        proj_x *= self.proj_w
        proj_y *= self.proj_h
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)
        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        depth = np.linalg.norm(xyz_proj, 2, axis=1)
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        indices = indices[order]
        depth = depth[order]
        xyz_proj = xyz_proj[order]
        intensity_proj = intensity_proj[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get range_representation of pointclouds
        proj_depth = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_xyz = np.full((self.proj_h, self.proj_w, 3), -1, dtype=np.float32)
        proj_intensity = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_mask = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)

        proj_depth[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = xyz_proj
        proj_intensity[proj_y, proj_x] = intensity_proj
        proj_idx[proj_y, proj_x] = indices
        proj_mask = (proj_idx > 0).astype(np.int32)

        proj_depth_tensor = torch.from_numpy(proj_depth).clone()
        proj_xyz_tensor = torch.from_numpy(proj_xyz).clone()
        proj_intensity_tensor = torch.from_numpy(proj_intensity).clone()
        proj_mask_tensor = torch.from_numpy(proj_mask)

        proj = torch.cat([
            proj_depth_tensor.unsqueeze(0).clone(),
            proj_xyz_tensor.clone().permute(2, 0, 1),
            proj_intensity_tensor.unsqueeze(0).clone()]
            )
        proj = (proj - self.range_img_means[:, None, None]) / self.range_img_stds[:, None, None]
        proj = proj * proj_mask_tensor.float()

        # reference_points
        voxel_centers_1_2 = self.voxel_centers_1_2.copy()
        voxel_centers_1_2_ego = self.inverse_lidar_to_ego_transform(voxel_centers_1_2, lidar2ego)

        proj_vox_x_1_2, proj_vox_y_1_2 = self.range_projection(voxel_centers_1_2_ego)
        mask1 = proj_vox_y_1_2 > 0
        mask2 = proj_vox_y_1_2 < 1
        mask_proj_y_1_2 = np.logical_and(mask1, mask2)
        proj_vox_centers_1_2 = np.concatenate((proj_vox_x_1_2.reshape(-1, 1), proj_vox_y_1_2.reshape(-1, 1)), axis=-1)
        proj_vox_centers_1_2 = torch.from_numpy(proj_vox_centers_1_2)               # (num_voxels, 2)
        proj_vox_centers_1_2 = proj_vox_centers_1_2.unsqueeze(0).permute(1, 0, 2)   # (1, num_voxels, 2) --> (num_voxels, 1, 2)
        proj_vox_centers_1_2 = proj_vox_centers_1_2.repeat(1, self.n_levels_1_2, 1) # (num_voxels, n_levels, 2), n_levels=4

        # mask for reference_points
        mask_1_2 = torch.from_numpy(mask_proj_y_1_2)

        # fine2coarse_mapping
        f2cmapping_1_2_tmp = self.f2cmapping_1_2.copy()
        f2cmapping_1_2 = torch.from_numpy(f2cmapping_1_2_tmp).long()

        data_dict = {}

        data_dict['xyz'] = xyz_raw
        data_dict['intensity'] = intensity_raw
        data_dict['grid_index'] = grid_index
        data_dict['voxel_coors'] = voxel_coors
        data_dict['point_index'] = point_index

        data_dict['proj'] = proj
        data_dict['proj_xy'] = proj_xy
        data_dict['proj_vox_centers_1_2'] = proj_vox_centers_1_2
        data_dict['mask_1_2'] = mask_1_2
        data_dict['f2cmapping_1_2'] = f2cmapping_1_2
        data_dict['voxel_label_1_1'] = voxel_label_1_1
        data_dict['voxel_label_1_2'] = voxel_label_1_2

        return data_dict
    

    def range_projection(self, points):
        points_copy = points.copy()
        depth = np.linalg.norm(points_copy, 2, axis=1)
        x = points_copy[:, 0]
        y = points_copy[:, 1]
        z = points_copy[:, 2]

        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        proj_x = 0.5 * (yaw / np.pi + 1.0)          
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v
        
        return proj_x, proj_y

    def get_vox_centers(self, grid_shape):
        xv, yv, zv = np.meshgrid(
            range(grid_shape[0]),
            range(grid_shape[1]),
            range(grid_shape[2]),
            indexing='ij'
            )
        vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
            ], axis=0).astype(int).T

        intervals = (self.max_volume_space - self.min_volume_space) / (grid_shape)
        voxel_centers = (vox_coords.astype(np.float32) + 0.5) * intervals + self.min_volume_space
        
        return voxel_centers

    @staticmethod
    def lidar_to_ego_transform(xyz, lidar2ego):
        w, x, y, z = lidar2ego['rotation']
        lidar2ego_rotation = Quaternion(w, x, y, z).rotation_matrix
        lidar2ego_translation = lidar2ego['translation']
        xyz_copy = xyz.copy()
        xyz_copy[:, :3] = xyz_copy[:, :3] @ lidar2ego_rotation.T
        xyz_copy[:, :3] += lidar2ego_translation
        return xyz_copy

    @staticmethod
    def inverse_lidar_to_ego_transform(xyz, lidar2ego):
        w, x, y, z = lidar2ego['rotation']
        lidar2ego_rotation = Quaternion(w, x, y, z).rotation_matrix
        lidar2ego_translation = lidar2ego['translation']
        xyz_copy = xyz.copy()
        xyz_copy[:, :3] -= lidar2ego_translation
        xyz_copy[:, :3] = xyz_copy[:, :3] @ lidar2ego_rotation
        return xyz_copy

    def computeFine2CoarseMapping(self, fine_voxel_centers, coarse_grid_shape, coarse_grid_size):
        fine_points = fine_voxel_centers.copy()
        fine_points[:, 0] -= self.min_volume_space[0]
        fine_points[:, 1] -= self.min_volume_space[1]
        fine_points[:, 2] -= self.min_volume_space[2]
        coarse_points = (fine_points / coarse_grid_size).astype(np.int32)
        coarse_index = (
            coarse_points[:, 0] * coarse_grid_shape[1] * coarse_grid_shape[2] 
            + coarse_points[:, 1] * coarse_grid_shape[2]
            + coarse_points[:, 2])
        
        return coarse_index


def collate_fn_default(data):
    
    xyz = [torch.from_numpy(d['xyz']).float() for d in data]
    intensity = [torch.from_numpy(d['intensity']).float() for d in data]
    grid_index_bs = [torch.from_numpy(d['grid_index']).float() for d in data]
    voxel_coors_bs = [torch.from_numpy(d['voxel_coors']).float() for d in data]

    batch_size = len(xyz)
    batch_idx = []
    for i in range(batch_size):
        batch_idx.append(torch.ones(len(xyz[i])) * i)
    batch_idx = torch.cat(batch_idx).reshape(-1, 1)

    voxel_coors_list = []
    coors_unq_list = []
    coors_inv_list = []
    num_scales = len(grid_index_bs[0])
    for i_scale in range(num_scales):
        grid_index_i_scale_list = []
        voxel_coors_i_scale_list = []
        for grid_index_batch, voxel_coors_batch in zip(grid_index_bs, voxel_coors_bs):
            grid_index_i_scale_list.append(grid_index_batch[i_scale])
            voxel_coors_i_scale_list.append(voxel_coors_batch[i_scale])
        grid_index_i_scale = torch.cat(grid_index_i_scale_list)
        voxel_coors_i_scale = torch.cat(voxel_coors_i_scale_list)
        voxel_coors_list.append(voxel_coors_i_scale)
        grid_index_i_scale = torch.cat((batch_idx, grid_index_i_scale), dim=1)
        coors_unq_i_scale, coors_inv_i_scale = torch.unique(grid_index_i_scale, return_inverse=True, dim=0)
        coors_unq_list.append(coors_unq_i_scale.to(torch.int32))
        coors_inv_list.append(coors_inv_i_scale.to(torch.int64))
    
    point_index = [torch.from_numpy(d['point_index']).float() for d in data]
    point_index = torch.cat(point_index)
    point_index = torch.cat((batch_idx, point_index), dim=1)
    point_coors_unq, point_coors_inv = torch.unique(point_index, return_inverse=True, dim=0)

    proj_list = [d['proj'] for d in data]
    proj_xy_list = [d['proj_xy'] for d in data]
    proj_vox_centers_1_2_list = [d['proj_vox_centers_1_2'] for d in data]
    mask_1_2_list = [d['mask_1_2'] for d in data]
    f2cmapping_1_2_list = [d['f2cmapping_1_2'] for d in data]
    voxel_label_1_1_list = [d['voxel_label_1_1'] for d in data]
    voxel_label_1_2_list = [d['voxel_label_1_2'] for d in data]

    return {
        'xyz': torch.cat(xyz).float(),
        'intensity': torch.cat(intensity).float(),
        'voxel_coors_list': voxel_coors_list,
        'coors_unq_list': coors_unq_list,
        'coors_inv_list': coors_inv_list,
        'point_coors_unq': point_coors_unq.to(torch.int32),
        'point_coors_inv': point_coors_inv.to(torch.int64),
        'batch_size': batch_size,
        'proj': torch.stack(proj_list, 0),
        'proj_xy': proj_xy_list,
        'proj_vox_centers_1_2': torch.stack(proj_vox_centers_1_2_list, 0),
        'mask_1_2': torch.stack(mask_1_2_list, 0),
        'f2cmapping_1_2': torch.stack(f2cmapping_1_2_list, 0),
        'voxel_label_1_1': torch.stack(voxel_label_1_1_list, 0),
        'voxel_label_1_2': torch.stack(voxel_label_1_2_list, 0),
    }