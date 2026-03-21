import os
import torch
import time
import datetime
import argparse
import sys
sys.path.append("../..")

from option import Option
from utils.recorder import Recorder
from utils.ddp_initialization import init_distributed_mode

## model
from train import Trainer


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings

        # set gpu
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.gpu
        init_distributed_mode(settings=self.settings)
        if self.settings.distributed:
            torch.distributed.barrier()
        
        # set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        if self.settings.distributed:
            torch.cuda.set_device(self.settings.gpu)
        else:
            torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

        # check save_path
        if not self.settings.distributed or (self.settings.rank == 0):
            self.settings.check_path()
            self.recorder = Recorder(config=self.settings.config, save_path=self.settings.save_path)    # make sure only the main process has recorder
        else:
            self.recorder = None

        # init trainer
        self.trainer = Trainer(settings=self.settings, recorder=self.recorder)
        self.epoch_start = 0

        # load checkpoint
        self._loadCheckpoint()


    def _loadCheckpoint(self):
        assert self.settings.pretrained_model is None or self.settings.checkpoint is None, "cannot use pretrained weight and checkpoint at the same time"
        ## pretrained_model full loading
        # if self.settings.pretrained_model is not None:
        #     if not os.path.isfile(self.settings.pretrained_model):
        #         raise FileNotFoundError("pretrained model not found: {}".format(
        #             self.settings.pretrained_model))
        #     load_state_dict = torch.load(self.settings.pretrained_model, map_location="cpu")
        #     model_state_dict = self.trainer.model.state_dict()
        #     for k, v in load_state_dict.items():
        #         if k in model_state_dict.keys():
        #             if model_state_dict[k].size() == v.size():
        #                 model_state_dict[k] = v
        #             else:
        #                 print("different size: ", k, v.size())
        #         else:
        #             print("different key: ", k)
        #     self.trainer.model.load_state_dict(load_state_dict)
        #     if self.recorder is not None:
        #         self.recorder.logger.info(
        #             "loading pretrained weight from: {}".format(self.settings.pretrained_model)
        #         )

        ## pretrained_model partial loading
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(
                    self.settings.pretrained_model))
            load_state_dict = torch.load(self.settings.pretrained_model, map_location="cpu")
            model_state_dict = self.trainer.model.state_dict()
            partial_load = {}
            for k, v in load_state_dict.items():
                if k in model_state_dict.keys() and "ssc_classifier" not in k:
                    if model_state_dict[k].size() == v.size():
                        partial_load[k] = v
                    else:
                        print("different size: ", k, v.size())
                else:
                    print("different key: ", k)
            model_state_dict.update(partial_load)
            self.trainer.model.load_state_dict(model_state_dict)
            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading pretrained weight from: {}".format(self.settings.pretrained_model)
                )

        # checkpoint
        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError("checkpoint file not found: {}".format(
                    self.settings.checkpoint))
            checkpoint = torch.load(self.settings.checkpoint, map_location="cpu")
            self.trainer.model.load_state_dict(checkpoint["model"])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            self.trainer.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch_start = checkpoint['epoch'] + 1
            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading checkpoints from: {}".format(self.settings.checkpoint)
                )


    def run(self):
        t_start = time.time()

        best_train_result = None
        best_val_result = None

        for epoch in range(self.epoch_start, self.settings.config['train_params']['max_num_epochs']):
            # train
            train_result = self.trainer.train(epoch)
            # if self.recorder is not None:
            #     if best_train_result is None:
            #         best_train_result = train_result
            #     for k, v in train_result.items():
            #         if v >= best_train_result[k]:
            #             self.recorder.logger.info(
            #                 "train: get better {} model: {}".format(k, v)
            #             )
            #             train_save_path = os.path.join(
            #                 self.recorder.checkpoint_path, "train_best_{}_model.pth".format(k)
            #             )
            #             best_train_result[k] = v
            #             torch.save(self.trainer.model.state_dict(), train_save_path)
            
            # valid
            if epoch % self.settings.config['train_params']['report_epochs'] == 0:
                val_result = self.trainer.valid(epoch)
                if self.recorder is not None:
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info(
                                "valid: get better {} model: {}".format(k, v)
                            )
                            val_save_path = os.path.join(
                                self.recorder.checkpoint_path, "val_best_{}_model.pth".format(k)
                            )
                            best_val_result[k] = v
                            torch.save(self.trainer.model.state_dict(), val_save_path)

            # save checkpoint
            if self.recorder is not None:
                checkpoint_save_path = os.path.join(
                    self.recorder.checkpoint_path, "checkpoint.pth"
                )              
                checkpoint_data = {
                    "model": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint_data, checkpoint_save_path)

                # log
                log_str = ">>> Best Val Result: "
                for k, v in best_val_result.items():
                    log_str += "{}: {}".format(k, v)
                self.recorder.logger.info(log_str)

        t_end = time.time()
        if self.recorder is not None:
            self.recorder.logger.info(
                "==== total cost time: {}".format(datetime.timedelta(seconds=(t_end - t_start)))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path to config file")
    args = parser.parse_args()

    settings = Option(config_path=args.config_path)
    experiment = Experiment(settings)

    print("===init env success===")
    experiment.run()