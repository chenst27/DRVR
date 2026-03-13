import numpy as np
import torch
import time
import yaml
import argparse
import logging
import tqdm
import os
import sys
sys.path.append("../..")

from models.drvr_smk import DRVR
from dataloader.dataset_semkitti_ssc import SemanticKITTI, collate_fn_default


# get configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="path to config file")
args = parser.parse_args()
config = yaml.safe_load(open(args.config_path, 'r'))
save_dir = os.path.join(config['infer_params']['save_dir'], "Evaluation")
os.makedirs(save_dir, exist_ok=True)

save_pred = config['infer_params']['save_pred']
if save_pred:
    pred_dir = os.path.join(save_dir, "testset")
    os.makedirs(pred_dir, exist_ok=True)

# load trained model
model = DRVR(config=config)
state_dict = torch.load(config['infer_params']['model_path'], map_location="cpu")
new_weights_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    new_weights_dict[new_k] = v
model.load_state_dict(new_weights_dict)
model.cuda()
model.eval()

# init data_loader
valid_dataset = SemanticKITTI(
    config=config, 
    dataset_type='test',
    flip_aug=False
)
data_files = valid_dataset.im_idx
cls_name = valid_dataset.class_names
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn_default
)

with open(valid_dataset.label_mapping, 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
maxkey = max(semkittiyaml['learning_map_inv'].keys())
inv_remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
inv_remap_lut[list(semkittiyaml['learning_map_inv'].keys())] = list(semkittiyaml['learning_map_inv'].values())

pbar = tqdm.tqdm(total=len(valid_loader))

for i, data_dict in enumerate(valid_loader):
    with torch.no_grad():
        ## cpu -> gpu
        data_dict['xyz'] = data_dict['xyz'].cuda()
        data_dict['intensity'] = data_dict['intensity'].cuda()
        data_dict['proj'] = data_dict['proj'].cuda()

        prediction = model(data_dict)
        ssc_logits_1_1 = prediction["ssc_logits_1_1"]
        pred_mask_1_1 = prediction['pred_mask_1_1']

        ssc_logits_eval = ssc_logits_1_1.clone()
        ssc_logits_eval = ssc_logits_eval[0].flatten(1).permute(1, 0)
        ssc_logits_eval = ssc_logits_eval.argmax(dim=1)
        ssc_logits_eval[~pred_mask_1_1] = 0

        if save_pred:
            frame_id = data_files[i][-10:-4]
            sequence = data_files[i][-22:-20]
            folder_dir = os.path.join(pred_dir, 'sequences', sequence, 'predictions')
            os.makedirs(folder_dir, exist_ok=True)
            pred_path = os.path.join(folder_dir, frame_id + '.label')
            ssc_logits_np = ssc_logits_eval.cpu().numpy()
            ssc_logits_np = ssc_logits_np.astype(np.uint16)
            ssc_logits_np = inv_remap_lut[ssc_logits_np].astype(np.uint16)
            ssc_logits_np.tofile(pred_path)

        pbar.update(1)