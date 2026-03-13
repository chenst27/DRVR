# train
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=59999 --use_env main.py \
--config_path /mnt/home/chensitao/code/DRVR/config/drvr_semantickitti.yaml \
2>&1 | tee /mnt/home/chensitao/code/DRVR/logs/smk.txt

# # eval (val-split)
# CUDA_VISIBLE_DEVICES="7" \
# python eval.py --config_path /mnt/home/chensitao/code/DRVR/config/drvr_semantickitti.yaml

# # test (test-split)
# CUDA_VISIBLE_DEVICES="7" \
# python test.py --config_path /mnt/home/chensitao/code/DRVR/config/drvr_semantickitti.yaml