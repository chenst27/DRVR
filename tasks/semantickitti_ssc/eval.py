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
from utils.iou_eval import IOUEval


def initLogger(log_dir):
    logger = logging.getLogger("console")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    file_handle = logging.FileHandler(os.path.join(log_dir, "infer_results.log"))
    file_handle.setFormatter(file_formatter)
    console_handle = logging.StreamHandler(sys.stdout)
    console_handle.setFormatter(console_formatter)
    logger.addHandler(file_handle)
    logger.addHandler(console_handle)
    return logger

# get configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="path to config file")
args = parser.parse_args()
config = yaml.safe_load(open(args.config_path, 'r'))
save_dir = os.path.join(config['infer_params']['save_dir'], "Evaluation")
os.makedirs(save_dir, exist_ok=True)

save_pred = config['infer_params']['save_pred']
if save_pred:
    pred_dir = os.path.join(save_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)

# init logger
log_dir = os.path.join(save_dir, "log")
os.makedirs(log_dir, exist_ok=True)
logger = initLogger(log_dir)

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

# init voxel metrics
ssc_iou_meter = IOUEval(
    n_classes=config['model_params']['num_classes'],
    device=torch.device("cpu"),
    is_distributed=False            
)
sc_iou_meter = IOUEval(
    n_classes=2,
    device=torch.device("cpu"),
    is_distributed=False         
)
ssc_iou_meter.reset()
sc_iou_meter.reset()

# init data_loader
valid_dataset = SemanticKITTI(
    config=config, 
    dataset_type='valid'
)
cls_name = valid_dataset.class_names
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn_default
)

pbar = tqdm.tqdm(total=len(valid_loader))

for i, data_dict in enumerate(valid_loader):
    with torch.no_grad():
        ## cpu -> gpu
        data_dict['xyz'] = data_dict['xyz'].cuda()
        data_dict['intensity'] = data_dict['intensity'].cuda()
        data_dict['proj'] = data_dict['proj'].cuda()
        data_dict['voxel_label_1_1'] = data_dict['voxel_label_1_1'].cuda()

        prediction = model(data_dict)
        ssc_logits_1_1 = prediction["ssc_logits_1_1"]
        pred_mask_1_1 = prediction['pred_mask_1_1']

        ssc_label_eval = data_dict['voxel_label_1_1'].clone()
        ssc_label_eval = ssc_label_eval.reshape(-1, 1)
        eval_mask = ssc_label_eval.lt(255).squeeze(1)
        ssc_label_eval = ssc_label_eval[eval_mask]

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
            ssc_logits_np = ssc_logits_np.astype(np.uint16).reshape((256, 256, 32))
            ssc_logits_np.tofile(pred_path)

        ssc_logits_eval = ssc_logits_eval[eval_mask]
        ssc_iou_meter.addBatch(ssc_logits_eval.long(), ssc_label_eval.long())

        sc_logits_eval = ssc_logits_eval.gt(0).long()
        sc_label_eval = ssc_label_eval.gt(0).long()
        sc_iou_meter.addBatch(sc_logits_eval, sc_label_eval)

        pbar.update(1)

_, ssc_class_iou = ssc_iou_meter.getIoU()
_, ssc_class_acc = ssc_iou_meter.getAcc()
_, ssc_class_recall = ssc_iou_meter.getRecall()
_, sc_iou = sc_iou_meter.getIoU()

log_str_end = ">>> Valid End: [ssc_miou_1_1:{:0.4f}] [sc_iou_1_1:{:0.4f}]".format(
                ssc_class_iou[1:].mean().item(), sc_iou[1].item()) 
logger.info(log_str_end)

for k, v in enumerate(cls_name):
    valid_log = "class:{} [SSC_IoU_1_1:{:0.4f}] [SSC_Acc_1_1:{:0.4f}] [SSC_Recall_1_1:{:0.4f}]".format(
                v, ssc_class_iou[k].item(), ssc_class_acc[k].item(), ssc_class_recall[k].item())
    logger.info(valid_log) 