import datetime

class RemainTime:
    def __init__(self, max_epochs, steps_per_epoch, steps_per_epoch_valid, data_time, batch_time_train, batch_time_valid):
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_epoch_valid = steps_per_epoch_valid
        # AverageMeter
        self.data_time = data_time
        self.batch_time_train = batch_time_train
        self.batch_time_valid = batch_time_valid


    def calculate(self, epoch, iter):
        remain_time_train = int((self.data_time.avg + self.batch_time_train.avg) * (self.steps_per_epoch * self.max_epochs - (epoch * self.steps_per_epoch + iter + 1)))
        remain_time_valid = int(self.batch_time_valid.avg * self.steps_per_epoch_valid * (self.max_epochs - epoch))

        remain_time = remain_time_train + remain_time_valid
        return str(datetime.timedelta(seconds=remain_time))
        