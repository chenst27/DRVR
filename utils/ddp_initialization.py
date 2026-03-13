import os
import torch
import torch.distributed as dist

# 使用torch.distributed.launch --use_env指令启动时，
# 会自动在python的os.environ中写入RANK、WORLD_SIZE、LOCAL_RANK信息。

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def init_distributed_mode(settings):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        settings.rank = int(os.environ["RANK"])
        settings.world_size = int(os.environ['WORLD_SIZE'])
        settings.gpu = int(os.environ['LOCAL_RANK'])
    elif "SLURM_PROCID" in os.environ:
        settings.rank = int(os.environ["SLURM_PROCID"])
        settings.gpu = settings.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        settings.distributed = False
        return
    settings.distributed = True
    settings.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        settings.rank, settings.dist_url), flush=True)
    dist.init_process_group(
        backend=settings.dist_backend, 
        init_method=settings.dist_url,
        world_size=settings.world_size, 
        rank=settings.rank)
