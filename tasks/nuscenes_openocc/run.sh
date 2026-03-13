# train
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=59999 --use_env main.py \
--config_path /mnt/home/chensitao/code/DRVR/config/drvr_nuscenes.yaml \
2>&1 | tee /mnt/home/chensitao/code/DRVR/logs/nus.txt

# # eval
# CUDA_VISIBLE_DEVICES="7" \
# python eval.py --config_path /mnt/home/chensitao/code/DRVR/config/drvr_nuscenes.yaml