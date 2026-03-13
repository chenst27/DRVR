import logging
import os
import sys
import torch
import torch.utils.tensorboard as tensorboard


class Recorder(object):
    def __init__(self, config, save_path, use_tensorboard=True):
        self.config = config
        self.save_path = save_path
        print('>> Init a recoder at ', self.save_path)

        self.log_path = os.path.join(self.save_path, "log")
        self.checkpoint_path = os.path.join(self.save_path, "checkpoint")
        self.code_path = os.path.join(self.save_path, "code")

        if use_tensorboard:
            self.tensorboard = tensorboard.SummaryWriter(log_dir=self.save_path)
        else:
            self.tensorboard = None
        
        # mkdir
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not os.path.isdir(self.code_path):
            os.makedirs(self.code_path)
        
        # init logger
        self.logger = self._initLogger()

        # save code_files
        self.save_codefiles()


    def _initLogger(self):
        logger = logging.getLogger("console")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')

        file_handle = logging.FileHandler(os.path.join(self.log_path, "console.log"))
        file_handle.setFormatter(file_formatter)

        console_handle = logging.StreamHandler(sys.stdout)
        console_handle.setFormatter(console_formatter)

        logger.addHandler(file_handle)
        logger.addHandler(console_handle)

        return logger
    

    # def save_checkpoint(self, to_save, suffix=""):
    #     torch.save(to_save, self.checkpoint_path + suffix)

    
    def save_codefiles(self):
        for i in range(len(self.config['save_params']['save_file'])):
            os.system('cp {} {}'.format(self.config['save_params']['save_file'][i], self.code_path))
