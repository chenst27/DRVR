import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("../..")

from option import Option
from models.drvr_smk import DRVR
from dataloader.dataset_semkitti_ssc import SemanticKITTI, collate_fn_default
from utils.warmupLR import WarmupCosineLR
from utils.iou_eval import IOUEval
from utils.avgmeter import AverageMeter
from utils.remain_time import RemainTime
from utils.dice_loss import InvertDiceLoss
from utils.focal_softmax import FocalSoftmaxLoss


class Trainer():
    def __init__(self, settings: Option, recorder=None):
        self.settings = settings
        self.config = settings.config
        self.recorder = recorder
        self.model = DRVR(config=self.config).cuda()
        self.num_classes = self.config['model_params']['num_classes']

        # init data_loader
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        # init voxel_class_weights
        cls_freq = np.array(self.config['dataset_params']['coarse_cls_freq'])
        cls_freq = cls_freq / cls_freq.sum()
        class_weights = np.log(1 + 1 / (cls_freq + 1e-8))
        self.coarse_class_weights = class_weights / class_weights.max()

        cls_freq = np.array(self.config['dataset_params']['fine_cls_freq'])
        cls_freq = cls_freq / cls_freq.sum()
        class_weights = np.log(1 + 1 / (cls_freq + 1e-8))
        self.fine_class_weights = class_weights / class_weights.max()

        # init criterion
        self.criterion = self._initCriterion_ssc()

        # init optimizer
        self.optimizer = self._initOptimizer()

        # init lr_scheduler
        self.scheduler = WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.config['train_params']["learning_rate"],
            warmup_steps=self.config['train_params']['warmup_epochs'] * len(self.train_loader),
            momentum=self.config['train_params']["lr_momentum"],
            max_steps=len(self.train_loader) * (self.config['train_params']['max_num_epochs'] - self.config['train_params']['warmup_epochs'])
            )

        # multi_gpu
        if self.settings.n_gpus > 1:
            if self.settings.distributed:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu], find_unused_parameters=True)
            else:
                self.model = nn.DataParallel(self.model)
                for k, v in self.criterion.items():
                    self.criterion[k] = nn.DataParallel(v).cuda()

        # init voxel metrics
        self.ssc_iou_meter = IOUEval(
            n_classes=self.config['model_params']['num_classes'],
            device=torch.device("cpu"),
            is_distributed=self.settings.distributed            
        )
        self.ssc_iou_meter.reset()

        self.sc_iou_meter = IOUEval(
            n_classes=2,
            device=torch.device("cpu"),
            is_distributed=self.settings.distributed            
        )
        self.sc_iou_meter.reset()

        # init remain_timer
        self.data_time = AverageMeter()
        self.batch_time_train = AverageMeter()
        self.batch_time_valid = AverageMeter()
        self.remain_time = RemainTime(
            max_epochs=self.config['train_params']['max_num_epochs'],
            steps_per_epoch=len(self.train_loader),
            steps_per_epoch_valid=len(self.val_loader),
            data_time=self.data_time,
            batch_time_train=self.batch_time_train,
            batch_time_valid=self.batch_time_valid
            )


    def _initDataloader(self):
        train_dataset = SemanticKITTI(config=self.config, dataset_type='train', flip_aug=self.config['dataset_params']['train_data_loader']['flip_aug'])
        valid_dataset = SemanticKITTI(config=self.config, dataset_type='valid', flip_aug=self.config['dataset_params']['val_data_loader']['flip_aug'])

        self.cls_name = train_dataset.class_names

        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=train_dataset,
                shuffle=True,
                drop_last=True
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=valid_dataset,
                shuffle=False,
                drop_last=False
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.config['dataset_params']['train_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['train_data_loader']['num_workers'],
                drop_last=True,
                collate_fn=collate_fn_default,
                sampler=train_sampler
            )
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=self.config['dataset_params']['val_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['val_data_loader']['num_workers'],
                drop_last=False,
                collate_fn=collate_fn_default,
                sampler=valid_sampler
            )
            return train_loader, valid_loader, train_sampler, valid_sampler
        
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.config['dataset_params']['train_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['train_data_loader']['num_workers'],
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn_default
            )
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=self.config['dataset_params']['val_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['val_data_loader']['num_workers'],
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn_default
            )
            return train_loader, valid_loader, None, None
    

    def _initOptimizer(self):
        if self.config['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                params=self.model.parameters(), 
                lr=self.config['train_params']["learning_rate"],
                weight_decay=self.config['train_params']["weight_decay"]
                )
        elif self.config['train_params']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                lr=self.config['train_params']["learning_rate"],
                weight_decay=self.config['train_params']["weight_decay"],
                amsgrad=True)
        elif self.config['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.config['train_params']["learning_rate"],
                momentum=self.config['train_params']["momentum"],
                weight_decay=self.config['train_params']["weight_decay"],
                nesterov=self.config['train_params']["nesterov"]
                )
        else:
            raise NotImplementedError

        return optimizer

    
    def _initCriterion_ssc(self):
        criterion = {}
        criterion["geo_dice_loss"] = InvertDiceLoss()
        criterion["geo_focal_loss"] = FocalSoftmaxLoss(
            2, gamma=2, softmax=False, alpha=self.coarse_class_weights)
        criterion["sem_dice_loss"] = InvertDiceLoss()
        criterion["sem_focal_loss"] = FocalSoftmaxLoss(
            self.config['model_params']['num_classes'], gamma=2, softmax=False, alpha=self.fine_class_weights)
        return criterion
        
    
    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def _computeClassifyLoss_ssc(self, pred, label):
        pred_softmax = F.softmax(pred, dim=1)
        loss_sem_focal = self.criterion["sem_focal_loss"](pred_softmax, label.long()) * self.config['train_params']['focal_weight']
        loss_sem_dice = self.criterion["sem_dice_loss"](pred_softmax, label.long()) * self.config['train_params']['dice_weight']
        return loss_sem_focal, loss_sem_dice
    

    def _computeClassifyLoss_sc(self, pred, label):
        pred_softmax = F.softmax(pred, dim=1)
        loss_geo_focal = self.criterion["geo_focal_loss"](pred_softmax, label.long()) * self.config['train_params']['focal_weight']
        loss_geo_dice = self.criterion["geo_dice_loss"](pred_softmax, label.long()) * self.config['train_params']['dice_weight']
        return loss_geo_focal, loss_geo_dice

    
    def train(self, epoch):
        if self.settings.distributed:
            torch.distributed.barrier()
            self.train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        
        self.model.train()
        self.ssc_iou_meter.reset()
        self.sc_iou_meter.reset()

        t_start = time.time()

        for i, data_dict in enumerate(self.train_loader):
            t_process_start = time.time()

            ## cpu -> gpu
            data_dict['xyz'] = data_dict['xyz'].cuda()
            data_dict['intensity'] = data_dict['intensity'].cuda()
            data_dict['proj'] = data_dict['proj'].cuda()
            voxel_label_1_1 = data_dict['voxel_label_1_1'].cuda()
            voxel_label_1_2 = data_dict['voxel_label_1_2'].cuda()

            prediction = self.model(data_dict)

            ssc_logits_1_1 = prediction["ssc_logits_1_1"]
            sc_logits_1_2 = prediction["sc_logits_1_2"]
            num_nonempty_1_1 = prediction['num_nonempty_1_1']
            num_nonempty_1_2 = prediction['num_nonempty_1_2']
            pred_mask_1_1 = prediction['pred_mask_1_1']

            # 1:2 geo
            voxel_label_1_2_loss = voxel_label_1_2.clone().view(-1, 1)
            label_mask_1_2 = (voxel_label_1_2_loss == 0) | (voxel_label_1_2_loss == 255)
            voxel_label_1_2_loss[~label_mask_1_2] = 1
            label_mask_1_2 = (voxel_label_1_2_loss != 255).reshape(-1)
            voxel_label_1_2_loss = voxel_label_1_2_loss[label_mask_1_2]

            sc_logits_1_2 = sc_logits_1_2.permute(0,2,3,4,1).view(-1, 2)

            loss_geo_focal_1_2, loss_geo_dice_1_2 = self._computeClassifyLoss_sc(
                sc_logits_1_2[label_mask_1_2], voxel_label_1_2_loss)

            # 1:1 geo
            voxel_label_1_1_loss = voxel_label_1_1.clone().view(-1, 1)
            voxel_label_1_1_loss[~pred_mask_1_1] = 255
            label_mask_1_1 = (voxel_label_1_1_loss == 0) | (voxel_label_1_1_loss == 255)
            voxel_label_1_1_loss[~label_mask_1_1] = 1
            label_mask_1_1 = (voxel_label_1_1_loss != 255).reshape(-1)
            voxel_label_1_1_loss = voxel_label_1_1_loss[label_mask_1_1]

            empty_idx = 0
            ssc_logits_1_1 = ssc_logits_1_1.permute(0,2,3,4,1).view(-1, self.num_classes)
            sc_logits_1_1 = torch.cat([
                ssc_logits_1_1[..., empty_idx:empty_idx+1], 
                ssc_logits_1_1[..., empty_idx+1:].max(dim=-1, keepdim=True)[0]], 
                dim=-1)

            loss_geo_focal_1_1, loss_geo_dice_1_1 = self._computeClassifyLoss_sc(
                sc_logits_1_1[label_mask_1_1], voxel_label_1_1_loss)

            # 1:2 sem
            voxel_label_1_1_loss = voxel_label_1_1.clone().view(-1, 1)
            voxel_label_1_1_loss[~pred_mask_1_1] = 255
            label_mask_1_1 = (voxel_label_1_1_loss != 255).reshape(-1)
            voxel_label_1_1_loss = voxel_label_1_1_loss[label_mask_1_1]

            loss_sem_focal_1_1, loss_sem_dice_1_1 = self._computeClassifyLoss_ssc(
                ssc_logits_1_1[label_mask_1_1], voxel_label_1_1_loss)
            
            # loss
            total_loss = 0
            total_loss += loss_geo_focal_1_2
            total_loss += loss_geo_dice_1_2
            total_loss += loss_geo_focal_1_1
            total_loss += loss_geo_dice_1_1
            total_loss += loss_sem_focal_1_1
            total_loss += loss_sem_dice_1_1

            self._backward(total_loss)
            self.scheduler.step()

            loss_meter.update(total_loss.item())
            loss = total_loss.mean()

            # ## scene completion results
            # ssc_label_eval = voxel_label_1_1.clone()
            # ssc_label_eval = ssc_label_eval.reshape(-1, 1)
            # eval_mask = ssc_label_eval.lt(255).squeeze(1)
            # ssc_label_eval = ssc_label_eval[eval_mask]

            # ssc_logits_eval = ssc_logits_1_1.clone()
            # ssc_logits_eval = ssc_logits_eval[0].flatten(1).permute(1, 0)
            # ssc_logits_eval = ssc_logits_eval.argmax(dim=1)
            # ssc_logits_eval[~pred_mask_1_1] = 0
            # ssc_logits_eval = ssc_logits_eval[eval_mask]
            # self.ssc_iou_meter.addBatch(ssc_logits_eval.long(), ssc_label_eval.long())
            # _, ssc_class_iou = self.ssc_iou_meter.getIoU()
            # _, ssc_class_acc = self.ssc_iou_meter.getAcc()
            # _, ssc_class_recall = self.ssc_iou_meter.getRecall()

            # sc_logits_eval = ssc_logits_eval.gt(0).long()
            # sc_label_eval = ssc_label_eval.gt(0).long()
            # self.sc_iou_meter.addBatch(sc_logits_eval, sc_label_eval)
            # _, sc_iou = self.sc_iou_meter.getIoU()
            # _, sc_acc = self.sc_iou_meter.getAcc()
            # _, sc_recall = self.sc_iou_meter.getRecall()

            self.data_time.update(t_process_start - t_start)
            self.batch_time_train.update(time.time() - t_process_start)
            t_start = time.time()
            estimated_time = self.remain_time.calculate(epoch, i)
            
            if self.recorder is not None:
                for g in self.optimizer.param_groups:
                    lr = g['lr']
                    break
                log_str = ">>> Train [Epoch:{:03d}/{:03d}] [Iter:{:04d}/{:04d}] [DT:{data_time.val:.3f} ({data_time.avg:.3f})] [PT:{batch_time.val:.3f} ({batch_time.avg:.3f})] ".format(
                            epoch+1, self.config['train_params']['max_num_epochs'], i+1, len(self.train_loader), data_time=self.data_time, batch_time=self.batch_time_train)
                log_str += "[LR:{:0.5f}] [Loss:{:0.4f}] [num_nonempty_1_1:{}] [num_nonempty_1_2:{}] [RT:{}]".format(
                            lr, loss.item(), num_nonempty_1_1, num_nonempty_1_2, estimated_time)
                # log_str += "[LR:{:0.5f}] [Loss:{:0.4f}] [ssc_miou_1_1:{:0.4f}] [ssc_macc_1_1:{:0.4f}] [ssc_mrecall_1_1:{:0.4f}] ".format(
                #             lr, loss.item(), ssc_class_iou[1:].mean().item(), ssc_class_acc[1:].mean().item(), ssc_class_recall[1:].mean().item()) 
                # log_str += "[sc_iou_1_1:{:0.4f}] [sc_acc_1_1:{:0.4f}] [sc_recall_1_1:{:0.4f}] [num_nonempty_1_1:{}] [num_nonempty_1_2:{}] [RT:{}]".format(
                #             sc_iou[1].item(), sc_acc[1].item(), sc_recall[1].item(), num_nonempty_1_1, num_nonempty_1_2, estimated_time)
                self.recorder.logger.info(log_str)
        
        if self.recorder is not None:
            # end of train
            log_str_end = ">>> Train End: [Loss:{:0.4f}]".format(loss_meter.avg) 
            # log_str_end = ">>> Train End: [Loss:{:0.4f}] [ssc_miou_1_1:{:0.4f}] [sc_iou_1_1:{:0.4f}]".format(
            #             loss_meter.avg, ssc_class_iou[1:].mean().item(), sc_iou[1].item())  
            self.recorder.logger.info(log_str_end)

            # for k, v in enumerate(self.cls_name):
            #     train_log = "class:{} [SSC_IoU_1_1:{:0.4f}] [SSC_Acc_1_1:{:0.4f}] [SSC_Recall_1_1:{:0.4f}]".format(
            #                 v, ssc_class_iou[k].item(), ssc_class_acc[k].item(), ssc_class_recall[k].item())
            #     self.recorder.logger.info(train_log) 
            #     self.recorder.tensorboard.add_scalar(
            #         tag="Train_{:02d}_{}_SSC_IoU_1_1".format(k, v), 
            #         scalar_value=ssc_class_iou[k].item(), global_step=epoch)
            #     self.recorder.tensorboard.add_scalar(
            #         tag="Train_{:02d}_{}_SSC_Acc_1_1".format(k, v), 
            #         scalar_value=ssc_class_acc[k].item(), global_step=epoch)
            #     self.recorder.tensorboard.add_scalar(
            #         tag="Train_{:02d}_{}_SSC_Recall_1_1".format(k, v), 
            #         scalar_value=ssc_class_recall[k].item(), global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="Train_Loss", scalar_value=loss_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="Train_lr", scalar_value=lr, global_step=epoch)
            # self.recorder.tensorboard.add_scalar(
            #     tag="Train_SC_IOU_1_1", scalar_value=sc_iou[1].item(), global_step=epoch)
            # self.recorder.tensorboard.add_scalar(
            #     tag="Train_SC_Acc_1_1", scalar_value=sc_acc[1].item(), global_step=epoch)
            # self.recorder.tensorboard.add_scalar(
            #     tag="Train_SC_Recall_1_1", scalar_value=sc_recall[1].item(), global_step=epoch)
            # self.recorder.tensorboard.add_scalar(
            #     tag="Train_SSC_MIOU_1_1", scalar_value=ssc_class_iou[1:].mean().item(), global_step=epoch)
            # self.recorder.tensorboard.add_scalar(
            #     tag="Train_SSC_MAcc_1_1", scalar_value=ssc_class_acc[1:].mean().item(), global_step=epoch)
            # self.recorder.tensorboard.add_scalar(
            #     tag="Train_SSC_MRecall_1_1", scalar_value=ssc_class_recall[1:].mean().item(), global_step=epoch)

        result_metrics = {}
        # result_metrics['train_sc_iou_1_1'] = sc_iou[1].item()
        # result_metrics['train_ssc_miou_1_1'] = ssc_class_iou[1:].mean().item()

        return result_metrics


    def valid(self, epoch):
        if self.settings.distributed:
            torch.distributed.barrier()
        
        loss_meter = AverageMeter()

        self.model.eval()
        self.ssc_iou_meter.reset()
        self.sc_iou_meter.reset()

        t_start = time.time()

        with torch.no_grad():
            for i, data_dict in enumerate(self.val_loader):
                
                ## cpu -> gpu
                data_dict['xyz'] = data_dict['xyz'].cuda()
                data_dict['intensity'] = data_dict['intensity'].cuda()
                data_dict['proj'] = data_dict['proj'].cuda()
                voxel_label_1_1 = data_dict['voxel_label_1_1'].cuda()
                voxel_label_1_2 = data_dict['voxel_label_1_2'].cuda()

                prediction = self.model(data_dict)

                ssc_logits_1_1 = prediction["ssc_logits_1_1"]
                sc_logits_1_2 = prediction["sc_logits_1_2"]
                num_nonempty_1_1 = prediction['num_nonempty_1_1']
                num_nonempty_1_2 = prediction['num_nonempty_1_2']
                pred_mask_1_1 = prediction['pred_mask_1_1']

                # 1:2 geo
                voxel_label_1_2_loss = voxel_label_1_2.clone().view(-1, 1)
                label_mask_1_2 = (voxel_label_1_2_loss == 0) | (voxel_label_1_2_loss == 255)
                voxel_label_1_2_loss[~label_mask_1_2] = 1
                label_mask_1_2 = (voxel_label_1_2_loss != 255).reshape(-1)
                voxel_label_1_2_loss = voxel_label_1_2_loss[label_mask_1_2]

                sc_logits_1_2 = sc_logits_1_2.permute(0,2,3,4,1).view(-1, 2)

                loss_geo_focal_1_2, loss_geo_dice_1_2 = self._computeClassifyLoss_sc(
                    sc_logits_1_2[label_mask_1_2], voxel_label_1_2_loss)

                # 1:1 geo
                voxel_label_1_1_loss = voxel_label_1_1.clone().view(-1, 1)
                voxel_label_1_1_loss[~pred_mask_1_1] = 255
                label_mask_1_1 = (voxel_label_1_1_loss == 0) | (voxel_label_1_1_loss == 255)
                voxel_label_1_1_loss[~label_mask_1_1] = 1
                label_mask_1_1 = (voxel_label_1_1_loss != 255).reshape(-1)
                voxel_label_1_1_loss = voxel_label_1_1_loss[label_mask_1_1]

                empty_idx = 0
                ssc_logits_1_1 = ssc_logits_1_1.permute(0,2,3,4,1).view(-1, self.num_classes)
                sc_logits_1_1 = torch.cat([
                    ssc_logits_1_1[..., empty_idx:empty_idx+1], 
                    ssc_logits_1_1[..., empty_idx+1:].max(dim=-1, keepdim=True)[0]], 
                    dim=-1)

                loss_geo_focal_1_1, loss_geo_dice_1_1 = self._computeClassifyLoss_sc(
                    sc_logits_1_1[label_mask_1_1], voxel_label_1_1_loss)

                # 1:2 sem
                voxel_label_1_1_loss = voxel_label_1_1.clone().view(-1, 1)
                voxel_label_1_1_loss[~pred_mask_1_1] = 255
                label_mask_1_1 = (voxel_label_1_1_loss != 255).reshape(-1)
                voxel_label_1_1_loss = voxel_label_1_1_loss[label_mask_1_1]

                loss_sem_focal_1_1, loss_sem_dice_1_1 = self._computeClassifyLoss_ssc(
                    ssc_logits_1_1[label_mask_1_1], voxel_label_1_1_loss)
                
                # loss
                total_loss = 0
                total_loss += loss_geo_focal_1_2
                total_loss += loss_geo_dice_1_2
                total_loss += loss_geo_focal_1_1
                total_loss += loss_geo_dice_1_1
                total_loss += loss_sem_focal_1_1
                total_loss += loss_sem_dice_1_1

                loss_meter.update(total_loss.item())
                loss = total_loss.mean()

                ## scene completion results
                ssc_label_eval = voxel_label_1_1.clone()
                ssc_label_eval = ssc_label_eval.reshape(-1, 1)
                eval_mask = ssc_label_eval.lt(255).squeeze(1)
                ssc_label_eval = ssc_label_eval[eval_mask]

                ssc_logits_eval = ssc_logits_1_1.clone()
                ssc_logits_eval = ssc_logits_eval.argmax(dim=1)
                ssc_logits_eval[~pred_mask_1_1] = 0
                ssc_logits_eval = ssc_logits_eval[eval_mask]
                self.ssc_iou_meter.addBatch(ssc_logits_eval.long(), ssc_label_eval.long())
                _, ssc_class_iou = self.ssc_iou_meter.getIoU()
                _, ssc_class_acc = self.ssc_iou_meter.getAcc()
                _, ssc_class_recall = self.ssc_iou_meter.getRecall()

                sc_logits_eval = ssc_logits_eval.gt(0).long()
                sc_label_eval = ssc_label_eval.gt(0).long()
                self.sc_iou_meter.addBatch(sc_logits_eval, sc_label_eval)
                _, sc_iou = self.sc_iou_meter.getIoU()
                _, sc_acc = self.sc_iou_meter.getAcc()
                _, sc_recall = self.sc_iou_meter.getRecall()

                self.batch_time_valid.update(time.time() - t_start)
                t_start = time.time()

                if self.recorder is not None:
                    log_str = ">>> Valid [Iter:{:04d}/{:04d}] [PT:{batch_time.val:.3f} ({batch_time.avg:.3f})] ".format(
                                i+1, len(self.val_loader), batch_time=self.batch_time_valid)
                    log_str += "[Loss:{:0.4f}] [ssc_miou_1_1:{:0.4f}] [ssc_macc_1_1:{:0.4f}] [ssc_mrecall_1_1:{:0.4f}] ".format(
                                loss.item(), ssc_class_iou[1:].mean().item(), ssc_class_acc[1:].mean().item(), ssc_class_recall[1:].mean().item()) 
                    log_str += "[sc_iou_1_1:{:0.4f}] [sc_acc_1_1:{:0.4f}] [sc_recall_1_1:{:0.4f}] [num_nonempty_1_1:{}] [num_nonempty_1_2:{}]".format(
                                sc_iou[1].item(), sc_acc[1].item(), sc_recall[1].item(), num_nonempty_1_1, num_nonempty_1_2)
                    self.recorder.logger.info(log_str)


            if self.recorder is not None:
                # end of valid 
                log_str_end = ">>> Valid End: [ssc_miou_1_1:{:0.4f}] [sc_iou_1_1:{:0.4f}]".format(
                                ssc_class_iou[1:].mean().item(), sc_iou[1].item()) 
                self.recorder.logger.info(log_str_end)

                for k, v in enumerate(self.cls_name):
                    valid_log = "class:{} [SSC_IoU_1_1:{:0.4f}] [SSC_Acc_1_1:{:0.4f}] [SSC_Recall_1_1:{:0.4f}]".format(
                                v, ssc_class_iou[k].item(), ssc_class_acc[k].item(), ssc_class_recall[k].item())
                    self.recorder.logger.info(valid_log) 
                    self.recorder.tensorboard.add_scalar(
                        tag="Valid_{:02d}_{}_SSC_IoU_1_1".format(k, v), 
                        scalar_value=ssc_class_iou[k].item(), global_step=epoch)
                    self.recorder.tensorboard.add_scalar(
                        tag="Valid_{:02d}_{}_SSC_Acc_1_1".format(k, v), 
                        scalar_value=ssc_class_acc[k].item(), global_step=epoch)
                    self.recorder.tensorboard.add_scalar(
                        tag="Valid_{:02d}_{}_SSC_Recall_1_1".format(k, v), 
                        scalar_value=ssc_class_recall[k].item(), global_step=epoch)

                # tensorboard
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_Loss", scalar_value=loss_meter.avg, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SC_IOU_1_1", scalar_value=sc_iou[1].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SC_Acc_1_1", scalar_value=sc_acc[1].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SC_Recall_1_1", scalar_value=sc_recall[1].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SSC_MIOU_1_1", scalar_value=ssc_class_iou[1:].mean().item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SSC_MAcc_1_1", scalar_value=ssc_class_acc[1:].mean().item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SSC_MRecall_1_1", scalar_value=ssc_class_recall[1:].mean().item(), global_step=epoch)

            result_metrics = {}
            result_metrics['valid_sc_iou_1_1'] = sc_iou[1].item()
            result_metrics['valid_ssc_miou_1_1'] = ssc_class_iou[1:].mean().item()

        return result_metrics