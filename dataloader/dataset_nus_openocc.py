import os
import glob
import numpy as np
import torch
from torch.utils import data
import yaml
import pickle
import numba as nb


class NuscenesOpenOccupancy(data.Dataset):
    def __init__(self, config, pkl_file, flip_aug=False):
        self.config = config
        self.flip_aug = flip_aug

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']

        self.data_path = self.config['dataset_params']['data_path']
        self.label_path = self.config['dataset_params']['label_path']
        self.grid_shape_list = np.asarray(self.config['dataset_params']['grid_shape_list'])
        self.max_volume_space = np.asarray(self.config['dataset_params']['max_volume_space'])
        self.min_volume_space = np.asarray(self.config['dataset_params']['min_volume_space'])

        self.sweeps_num = self.config['dataset_params']['sweeps_num']

        self.proj_h = self.config['dataset_params']['proj_img_size'][0]
        self.proj_w = self.config['dataset_params']['proj_img_size'][1]
        self.range_img_means = torch.tensor(self.config['dataset_params']["range_img_means"], dtype=torch.float)
        self.range_img_stds = torch.tensor(self.config['dataset_params']["range_img_stds"], dtype=torch.float) 

        self.fov_up = self.config['dataset_params']['fov_up'] / 180.0 * np.pi
        self.fov_down = self.config['dataset_params']['fov_down'] / 180.0 * np.pi
        self.fov_v = abs(self.fov_up) + abs(self.fov_down)

        self.class_names = [ 'free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',  
                             'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',  
                             'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']

        # get reference_points for deformable_transformer
        voxel_centers_1_4 = self.get_vox_centers(grid_shape=[128, 128, 10])
        proj_vox_x_1_4, proj_vox_y_1_4 = self.range_projection(voxel_centers_1_4)
        mask1 = proj_vox_y_1_4 > 0
        mask2 = proj_vox_y_1_4 < 1
        mask_proj_y_1_4 = np.logical_and(mask1, mask2)
        proj_vox_y_1_4 = np.clip(proj_vox_y_1_4, 0.01, 0.99)
        self.proj_vox_centers_1_4 = np.concatenate((proj_vox_x_1_4.reshape(-1, 1), proj_vox_y_1_4.reshape(-1, 1)), axis=-1)
        self.mask_proj_y_1_4 = mask_proj_y_1_4

        self.n_levels_1_4 = self.config['model_params']['cross_attn_n_levels_1_4']

        # get mapping
        voxel_centers_1_1 = self.get_vox_centers(grid_shape=[512, 512, 40])
        self.f2cmapping_1_2 = self.computeFine2CoarseMapping(
            fine_voxel_centers=voxel_centers_1_1, 
            coarse_grid_shape=[256, 256, 20], coarse_grid_size=0.4
        )
        voxel_centers_1_2 = self.get_vox_centers(grid_shape=[256, 256, 20])
        self.f2cmapping_2_4 = self.computeFine2CoarseMapping(
            fine_voxel_centers=voxel_centers_1_2, 
            coarse_grid_shape=[128, 128, 10], coarse_grid_size=0.8
        )

        self.grid_shape = np.array([128, 128, 10])


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)


    def __getitem__(self, index):

        info = self.nusc_infos[index]

        # get occupancy label
        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(info['scene_token'], info['lidar_token'])
        label = np.load(os.path.join(self.label_path, rel_path))  #  [z y x cls]
        occ_label = label[..., -1:]
        occ_label[occ_label==0] = 255       # noise --> 255
        occ_xyz_grid = label[..., [2,1,0]]  # z y x  -->  x y z
        label_voxel_pair = np.concatenate([occ_xyz_grid, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid[:, 0], occ_xyz_grid[:, 1], occ_xyz_grid[:, 2])), :].astype(np.int32)
        voxel_label_1_1 = np.zeros([512, 512, 40], dtype=np.uint8)
        voxel_label_1_1 = nb_process_label(voxel_label_1_1, label_voxel_pair)

        # get pointclouds
        lidar_path = info['lidar_path']
        lidar_path = lidar_path.replace("./data/nuscenes/", self.data_path)
        points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])

        xyz_proj = points[:, :3].copy()
        intensity_proj = points[:, 3].copy()

        if self.sweeps_num > 0:
            sweep_points_list = [points]
            if len(info['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(info['sweeps']))
            else:
                choices = np.random.choice(len(info['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = info['sweeps'][idx]
                sweep_path = sweep['data_path'].replace("./data/nuscenes/", self.data_path)
                points_sweep = np.fromfile(sweep_path, dtype=np.float32, count=-1).reshape([-1, 5])
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                sweep_points_list.append(points_sweep)
            points = np.concatenate(sweep_points_list, axis=0)

        xyz = points[:, :3]
        intensity = points[:, 3]

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
        xyz = xyz[mask]
        intensity = intensity[mask]
        
        if self.flip_aug and np.random.rand() > 0.5:
            xyz[:, 1] = -xyz[:, 1]
            xyz_proj[:, 1] = -xyz_proj[:, 1]
            voxel_label_1_1_tmp = voxel_label_1_1.copy()
            for y in range(voxel_label_1_1_tmp.shape[1]):
                voxel_label_1_1_tmp[:, y, :] = voxel_label_1_1[:, voxel_label_1_1_tmp.shape[1]-1-y, :]
            voxel_label_1_1 = voxel_label_1_1_tmp
        
        if self.flip_aug and np.random.rand() > 0.5:
            xyz[:, 0] = -xyz[:, 0]
            xyz_proj[:, 0] = -xyz_proj[:, 0]
            voxel_label_1_1_tmp = voxel_label_1_1.copy()
            for x in range(voxel_label_1_1_tmp.shape[0]):
                voxel_label_1_1_tmp[x, :, :] = voxel_label_1_1[voxel_label_1_1_tmp.shape[0]-1-x, :, :]
            voxel_label_1_1 = voxel_label_1_1_tmp

        voxel_label_1_1 = torch.from_numpy(voxel_label_1_1).float()

        voxel_label_1_2 = voxel_label_1_1.clone().reshape(256, 2, 256, 2, 20, 2).permute(0,2,4,1,3,5).reshape(256, 256, 20, 8)
        empty_mask_1_2 = voxel_label_1_2.sum(-1) == 0
        voxel_label_1_2 = voxel_label_1_2.to(torch.int64)
        occ_space_1_2 = voxel_label_1_2[~empty_mask_1_2]
        occ_space_1_2[occ_space_1_2==0] = -torch.arange(len(occ_space_1_2[occ_space_1_2==0])).to(occ_space_1_2.device) - 1
        voxel_label_1_2[~empty_mask_1_2] = occ_space_1_2
        voxel_label_1_2 = torch.mode(voxel_label_1_2, dim=-1)[0]
        voxel_label_1_2[voxel_label_1_2<0] = 255
        voxel_label_1_2 = voxel_label_1_2.long()

        voxel_label_1_4 = voxel_label_1_1.clone().reshape(128, 4, 128, 4, 10, 4).permute(0,2,4,1,3,5).reshape(128, 128, 10, 64)
        empty_mask_1_4 = voxel_label_1_4.sum(-1) == 0
        voxel_label_1_4 = voxel_label_1_4.to(torch.int64)
        occ_space_1_4 = voxel_label_1_4[~empty_mask_1_4]
        occ_space_1_4[occ_space_1_4==0] = -torch.arange(len(occ_space_1_4[occ_space_1_4==0])).to(occ_space_1_4.device) - 1
        voxel_label_1_4[~empty_mask_1_4] = occ_space_1_4
        voxel_label_1_4 = torch.mode(voxel_label_1_4, dim=-1)[0]
        voxel_label_1_4[voxel_label_1_4<0] = 255
        voxel_label_1_4 = voxel_label_1_4.long()

        xyz_raw = xyz.copy()
        intensity_raw = intensity.copy()

        grid_index_list = []
        voxel_coors_list = []
        for idx in range(len(self.grid_shape_list)):
            intervals = (self.max_volume_space - self.min_volume_space) / (self.grid_shape_list[idx])
            if (intervals == 0).any(): print("Zero interval!")
            grid_index_ = (np.floor((xyz.copy() - self.min_volume_space) / intervals)).astype(np.int32)
            voxel_coors = (grid_index_.astype(np.float32) + 0.5) * intervals + self.min_volume_space
            grid_index_list.append(grid_index_)
            voxel_coors_list.append(voxel_coors)
        grid_index = np.stack(grid_index_list, axis=0)
        voxel_coors = np.stack(voxel_coors_list, axis=0)

        intervals = (self.max_volume_space - self.min_volume_space) / (self.grid_shape)
        if (intervals == 0).any(): print("Zero interval!")
        point_index = (np.floor((xyz.copy() - self.min_volume_space) / intervals)).astype(np.int32)

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
        proj_vox_centers_1_4_tmp = self.proj_vox_centers_1_4.copy()
        proj_vox_centers_1_4 = torch.from_numpy(proj_vox_centers_1_4_tmp)           # (num_voxels, 2)
        proj_vox_centers_1_4 = proj_vox_centers_1_4.unsqueeze(0).permute(1, 0, 2)   # (1, num_voxels, 2) --> (num_voxels, 1, 2)
        proj_vox_centers_1_4 = proj_vox_centers_1_4.repeat(1, self.n_levels_1_4, 1) # (num_voxels, n_levels, 2), n_levels=4

        # mask for reference_points
        mask_1_4_tmp = self.mask_proj_y_1_4.copy()
        mask_1_4 = torch.from_numpy(mask_1_4_tmp)

        # fine2coarse_mapping
        f2cmapping_1_2_tmp = self.f2cmapping_1_2.copy()
        f2cmapping_1_2 = torch.from_numpy(f2cmapping_1_2_tmp).long()
        f2cmapping_2_4_tmp = self.f2cmapping_2_4.copy()
        f2cmapping_2_4 = torch.from_numpy(f2cmapping_2_4_tmp).long()

        data_dict = {}

        data_dict['xyz'] = xyz_raw
        data_dict['intensity'] = intensity_raw
        data_dict['grid_index'] = grid_index
        data_dict['voxel_coors'] = voxel_coors
        data_dict['point_index'] = point_index

        data_dict['proj'] = proj
        data_dict['proj_xy'] = proj_xy
        data_dict['proj_vox_centers_1_4'] = proj_vox_centers_1_4
        data_dict['mask_1_4'] = mask_1_4
        data_dict['f2cmapping_1_2'] = f2cmapping_1_2
        data_dict['f2cmapping_2_4'] = f2cmapping_2_4
        data_dict['voxel_label_1_1'] = voxel_label_1_1
        data_dict['voxel_label_1_2'] = voxel_label_1_2
        data_dict['voxel_label_1_4'] = voxel_label_1_4

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
    proj_vox_centers_1_4_list = [d['proj_vox_centers_1_4'] for d in data]
    mask_1_4_list = [d['mask_1_4'] for d in data]
    f2cmapping_1_2_list = [d['f2cmapping_1_2'] for d in data]
    f2cmapping_2_4_list = [d['f2cmapping_2_4'] for d in data]
    voxel_label_1_1_list = [d['voxel_label_1_1'] for d in data]
    voxel_label_1_2_list = [d['voxel_label_1_2'] for d in data]
    voxel_label_1_4_list = [d['voxel_label_1_4'] for d in data]

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
        'proj_vox_centers_1_4': torch.stack(proj_vox_centers_1_4_list, 0),
        'mask_1_4': torch.stack(mask_1_4_list, 0),
        'f2cmapping_1_2': torch.stack(f2cmapping_1_2_list, 0),
        'f2cmapping_2_4': torch.stack(f2cmapping_2_4_list, 0),
        'voxel_label_1_1': torch.stack(voxel_label_1_1_list, 0),
        'voxel_label_1_2': torch.stack(voxel_label_1_2_list, 0),
        'voxel_label_1_4': torch.stack(voxel_label_1_4_list, 0),
    }


@nb.jit('u1[:,:,:](u1[:,:,:],i4[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label