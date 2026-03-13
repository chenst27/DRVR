import os, glob, yaml
import numpy as np
from tqdm import tqdm
import numpy.matlib


def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1
  return uncompressed

def _read_SemKITTI(path, dtype, do_unpack):
  bin = np.fromfile(path, dtype=dtype)  # Flattened array
  if do_unpack:
    bin = unpack(bin)
  return bin

def _read_label_SemKITTI(path):
  label = _read_SemKITTI(path, dtype=np.uint16, do_unpack=False).astype(np.float32)
  return label

def _read_invalid_SemKITTI(path):
  invalid = _read_SemKITTI(path, dtype=np.uint8, do_unpack=True)
  return invalid

def get_remap_lut(path):
  dataset_config = yaml.safe_load(open(path, 'r'))
  maxkey = max(dataset_config['learning_map'].keys())
  remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
  remap_lut[list(dataset_config['learning_map'].keys())] = list(dataset_config['learning_map'].values())
  remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
  remap_lut[0] = 0  # only 'empty' stays 'empty'.
  return remap_lut

def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4):
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size
        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


if __name__ == "__main__":
    
    kitti_root = "/mnt/dataset/semantic-kitti"
    kitti_preprocess_root = "/mnt/dataset/semantic-kitti-ssc-preprocessed"
    downscaling = {"1_1": 1, "1_2": 2}
    scene_size = (256, 256, 32)
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    
    yaml_path = "/mnt/home/chensitao/code/DRVR/config/label_mapping/semantic-kitti.yaml"
    remap_lut = get_remap_lut(yaml_path)

    for sequence in sequences:
        sequence_path = os.path.join(kitti_root, "dataset", "sequences", sequence)
        label_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.label")))
        invalid_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.invalid")))
        out_dir = os.path.join(kitti_preprocess_root, "labels", sequence)
        os.makedirs(out_dir, exist_ok=True)

        for i in tqdm(range(len(label_paths))):
            frame_id, extension = os.path.splitext(os.path.basename(label_paths[i]))
            LABEL = _read_label_SemKITTI(label_paths[i])
            INVALID = _read_invalid_SemKITTI(invalid_paths[i])
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)
            LABEL[np.isclose(INVALID, 1)] = 255
            LABEL = LABEL.reshape([256, 256, 32])

            for scale in downscaling:
                filename = frame_id + "_" + scale + ".npy"
                label_filename = os.path.join(out_dir, filename)
                LABEL_ds = _downsample_label(LABEL, scene_size, downscaling[scale])
                np.save(label_filename, LABEL_ds)