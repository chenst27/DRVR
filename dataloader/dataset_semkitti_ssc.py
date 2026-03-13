import os
import glob
import numpy as np
import torch
from torch.utils import data
import yaml


class SemanticKITTI(data.Dataset):
    def __init__(self, config, dataset_type='train', flip_aug=False):
        self.config = config
        self.dataset_type = dataset_type
        self.flip_aug = flip_aug

        self.data_path = self.config['dataset_params']['data_path']
        self.label_path = self.config['dataset_params']['label_path']
        self.label_mapping = self.config['dataset_params']['label_mapping']
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

        with open(self.label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.mapped_class_name = semkittiyaml['mapped_class_name']
        self.learning_map = semkittiyaml['learning_map']
        if self.dataset_type == 'train':
            split = semkittiyaml['split']['train']
            data_set = 'semantic-kitti-ssc-all'
        elif self.dataset_type == 'valid':
            split = semkittiyaml['split']['valid']
            data_set = 'semantic-kitti-ssc-all'
        elif self.dataset_type == 'test':
            split = semkittiyaml['split']['test']
            data_set = 'semantic-kitti-ssc'
        else:
            raise Exception('Split must be train/val/test')
        
        self.class_names = [ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign"]

        self.im_idx = []
        for i_folder in split:
            # ../voxels/*bin
            files = sorted(glob.glob(os.path.join(self.data_path.replace('semantic-kitti', data_set), str(i_folder).zfill(2), 'voxels', '*.bin')))
            for filename in files:
                # ../velodyne/*bin
                self.im_idx.append(filename.replace(data_set, 'semantic-kitti').replace('voxels', 'velodyne'))
        
        # get reference_points for deformable_transformer
        voxel_centers_1_2 = self.get_vox_centers(grid_shape=[128, 128, 16])
        proj_vox_x_1_2, proj_vox_y_1_2 = self.range_projection(voxel_centers_1_2)
        mask1 = proj_vox_y_1_2 > 0
        mask2 = proj_vox_y_1_2 < 1
        mask_proj_y_1_2 = np.logical_and(mask1, mask2)
        proj_vox_y_1_2 = np.clip(proj_vox_y_1_2, 0.01, 0.99)
        self.proj_vox_centers_1_2 = np.concatenate((proj_vox_x_1_2.reshape(-1, 1), proj_vox_y_1_2.reshape(-1, 1)), axis=-1)
        self.mask_proj_y_1_2 = mask_proj_y_1_2

        self.n_levels_1_2 = self.config['model_params']['cross_attn_n_levels_1_2']

        # get mapping
        voxel_centers_1_1 = self.get_vox_centers(grid_shape=[256, 256, 32])
        self.f2cmapping_1_2 = self.computeFine2CoarseMapping(
            fine_voxel_centers=voxel_centers_1_1, 
            coarse_grid_shape=[128, 128, 16], coarse_grid_size=0.4
        )

        self.grid_shape = np.array([128, 128, 16])


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)


    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        xyz = raw_data[:, :3]
        intensity = raw_data[:, 3]

        # annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
        #                             dtype=np.uint32)
        # instance_label = annotated_data >> 16
        # annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

        # keep point in evaluation_range
        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
        xyz = xyz[mask]
        intensity = intensity[mask]
        # annotated_data = annotated_data[mask]
        
        # get voxel_label
        frame_id = self.im_idx[index][-10:-4]
        sequence = self.im_idx[index][-22:-20]
        if self.dataset_type == 'test':
            voxel_label_1_1 = np.zeros([256, 256, 32], dtype=int)
            voxel_label_1_2 = np.zeros([128, 128, 16], dtype=int)
        else:
            voxel_label_1_1_path = os.path.join(self.label_path, sequence, frame_id + "_1_1.npy")
            voxel_label_1_1 = np.load(voxel_label_1_1_path)
            voxel_label_1_2_path = os.path.join(self.label_path, sequence, frame_id + "_1_2.npy")
            voxel_label_1_2 = np.load(voxel_label_1_2_path)
        
        if self.flip_aug and np.random.rand() > 0.5:
            xyz[:, 1] = -xyz[:, 1]
            voxel_label_1_1_tmp = voxel_label_1_1.copy()
            for y in range(voxel_label_1_1_tmp.shape[1]):
                voxel_label_1_1_tmp[:, y, :] = voxel_label_1_1[:, voxel_label_1_1_tmp.shape[1]-1-y, :]
            voxel_label_1_2_tmp = voxel_label_1_2.copy()
            for y in range(voxel_label_1_2_tmp.shape[1]):
                voxel_label_1_2_tmp[:, y, :] = voxel_label_1_2[:, voxel_label_1_2_tmp.shape[1]-1-y, :]
            voxel_label_1_1 = voxel_label_1_1_tmp
            voxel_label_1_2 = voxel_label_1_2_tmp
            
        voxel_label_1_1 = torch.from_numpy(voxel_label_1_1).float()
        voxel_label_1_2 = torch.from_numpy(voxel_label_1_2).float()

        xyz_raw = xyz.copy()
        intensity_raw = intensity.copy()

        grid_index_list = []
        voxel_coors_list = []
        for idx in range(len(self.grid_shape_list)):
            intervals = (self.max_volume_space - self.min_volume_space) / (self.grid_shape_list[idx])
            if (intervals == 0).any(): print("Zero interval!")
            grid_index_ = (np.floor((xyz.copy() - self.min_volume_space) / intervals)).astype(np.int32)
            voxel_coors_ = (grid_index_.astype(np.float32) + 0.5) * intervals + self.min_volume_space
            grid_index_list.append(grid_index_)
            voxel_coors_list.append(voxel_coors_)
        grid_index = np.stack(grid_index_list, axis=0)
        voxel_coors = np.stack(voxel_coors_list, axis=0)

        intervals = (self.max_volume_space - self.min_volume_space) / (self.grid_shape)
        if (intervals == 0).any(): print("Zero interval!")
        point_index = (np.floor((xyz.copy() - self.min_volume_space) / intervals)).astype(np.int32)

        # perform range_projection
        proj_x, proj_y = self.range_projection(xyz)

        proj_x *= self.proj_w
        proj_y *= self.proj_h
        proj_x = np.maximum(np.minimum(
            self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)
        proj_y = np.maximum(np.minimum(
            self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)

        proj_xy = np.concatenate((proj_x.copy().reshape(-1,1), proj_y.copy().reshape(-1,1)), axis=1)
        proj_xy = torch.from_numpy(proj_xy)

        depth = np.linalg.norm(xyz, 2, axis=1)

        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        indices = indices[order]
        depth = depth[order]
        xyz = xyz[order]
        intensity = intensity[order]
        # annotated_data = annotated_data[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get range_representation of pointclouds
        proj_depth = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_xyz = np.full((self.proj_h, self.proj_w, 3), -1, dtype=np.float32)
        proj_intensity = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_mask = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)
        # proj_label = np.zeros((self.proj_h, self.proj_w), dtype=np.int32)

        proj_depth[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = xyz
        proj_intensity[proj_y, proj_x] = intensity
        proj_idx[proj_y, proj_x] = indices
        proj_mask = (proj_idx > 0).astype(np.int32)
        # proj_label[proj_y, proj_x] = annotated_data

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

        # for SemnaticKITTI
        left_x = self.proj_w / 4
        right_x = self.proj_w * 3 / 4
        proj_new = proj[:, :, int(left_x):int(right_x)]

        # reference_points
        proj_vox_centers_1_2_tmp = self.proj_vox_centers_1_2.copy()
        proj_vox_centers_1_2 = torch.from_numpy(proj_vox_centers_1_2_tmp)           # (num_voxels, 2)
        proj_vox_centers_1_2 = proj_vox_centers_1_2.unsqueeze(0).permute(1, 0, 2)   # (1, num_voxels, 2) --> (num_voxels, 1, 2)
        proj_vox_centers_1_2 = proj_vox_centers_1_2.repeat(1, self.n_levels_1_2, 1) # (num_voxels, n_levels, 2), n_levels=4
        
        # mask for reference_points
        mask_1_2_tmp = self.mask_proj_y_1_2.copy()
        mask_1_2 = torch.from_numpy(mask_1_2_tmp)

        # fine2coarse_mapping
        f2cmapping_1_2_tmp = self.f2cmapping_1_2.copy()
        f2cmapping_1_2 = torch.from_numpy(f2cmapping_1_2_tmp).long()

        data_dict = {}

        data_dict['xyz'] = xyz_raw
        data_dict['intensity'] = intensity_raw
        data_dict['grid_index'] = grid_index
        data_dict['voxel_coors'] = voxel_coors
        data_dict['point_index'] = point_index

        data_dict['proj'] = proj_new
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