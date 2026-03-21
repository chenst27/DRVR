import yaml
import os
import shutil
import sys
sys.path.append("../..")
from utils.ddp_initialization import is_main_process


class Option(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(self.config_path, "r"))
      
        # DDP params
        self.save_path = self.config['save_params']['save_path']
        self.seed = self.config['common_params']['seed']
        self.gpu = self.config['common_params']['gpu']
        self.n_gpus = len(self.gpu.split(","))
        self.rank = 0
        self.world_size = 1
        self.distributed = False
        self.dist_backend = "nccl"
        self.dist_url = "env://"

        # checkpoint
        self.checkpoint = self.config['common_params']['checkpoint']
        self.pretrained_model = self.config['common_params']['pretrained_model']

        self._initialize()


    def _initialize(self):
        # check settings
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            batch_size = self.config['dataset_params']['train_data_loader']['batch_size'] * self.n_gpus
        else:
            batch_size = self.config['dataset_params']['train_data_loader']['batch_size']

        # folder name: log_dataset_nettype-batchsize-lr
        self.save_path = os.path.join(self.save_path, "log_{}_{}_bs{}_lr{}_{}".format(
            self.config['dataset_params']['dataset_name'], 
            self.config['model_params']['net_type'], 
            batch_size, 
            self.config['train_params']['learning_rate'], 
            self.config['common_params']['experiment_id'])
            )


    def check_path(self):
        if is_main_process():
            if os.path.exists(self.save_path):
                print("file exist: {}".format(self.save_path))
                action = input("Select Action: d(delete) / q(quit): ").lower().strip()
                if action == "d":
                    shutil.rmtree(self.save_path)
                else:
                    raise OSError("Directory exits: {}".format(self.save_path))

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)