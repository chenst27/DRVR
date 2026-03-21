## 🛠️Environment

1. Create the conda environment
```setup
conda create -n occ python=3.7
conda activate occ
```

2. Install the following dependencies
```setup
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.1/index.html
pip install spconv-cu116 tensorboard numba nuscenes-devkit
```

## 📦 Prepare Dataset

1. SemanticKITTI
    - Download SemanticKITTI from [SemanticKITTI](https://semantic-kitti.org/dataset.html)
    - Download SSC labels from [SemKitti-SSC](https://semantic-kitti.org/assets/data_odometry_voxels_all.zip)
    - Modify `utils/preprocess.py` and execute it to preprocess SSC labels.

2. SemanticPOSS
    - Download SemanticPOSS from [SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html)
    - Download SSC labels from [SemPoss-SSC](https://drive.google.com/file/d/1AGagbRwQe3aR8liaC4YnkMW1iwSCLvvN/view)

3. nuScenes-Occupancy
    - Download nuScenes from [nuScenes](https://www.nuscenes.org/nuscenes)
    - Download OCC labels from [nuScenes-Occupancy](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view)
    - Download the generated train info file from [Google Drive](https://github.com/JeffWang987/OpenOccupancy/releases/download/train_pkl/nuscenes_occ_infos_train.pkl).
    - Download the generated val info file from [Google Drive](https://github.com/JeffWang987/OpenOccupancy/releases/download/val_pkl/nuscenes_occ_infos_val.pkl).

4. Occ3D-nuScenes
    - Download nuScenes from [nuScenes](https://www.nuscenes.org/nuscenes)
    - Download OCC labels from [Occ3D-nuScenes](https://tsinghua-mars-lab.github.io/Occ3D/)

## 🎇 Training and Evaluation

1. SemanticKITTI

    - Modify the configuration file `config/drvr_semantickitti.yaml`.
    - Enter the `tasks/semantickitti_ssc` folder, modify `run.sh` and run the command `./run.sh`.

2. SemanticPOSS
    - Modify the configuration file `config/drvr_semanticposs.yaml`.
    - Enter the `tasks/semanticposs_ssc` folder, modify `run.sh` and run the command `./run.sh`.

3. nuScenes-Occupancy
    - Modify the configuration file `config/drvr_nuscenes.yaml`.
    - Enter the `tasks/nuscenes_openocc` folder, modify `run.sh` and run the command `./run.sh`.

4. Occ3D-nuScenes
    - Modify the configuration file `config/drvr_occ3d.yaml`.
    - Enter the `tasks/nuscenes_occ3d` folder, modify `run.sh` and run the command `./run.sh`.