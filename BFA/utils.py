import os, sys, time, random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch import nn
import torch
from models.quantization import quan_Conv2d, quan_Linear


def piecewise_clustering(var, lambda_coeff, l_norm):
    var1=(var[var.ge(0)]-var[var.ge(0)].mean()).pow(l_norm).sum()
    var2=(var[var.le(0)]-var[var.le(0)].mean()).pow(l_norm).sum()
    return lambda_coeff*(var1+var2)


def clustering_loss(model, lambda_coeff, l_norm=2):
    
    pc_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            pc_loss += piecewise_clustering(m.weight, lambda_coeff, l_norm)
    
    return pc_loss 


def change_quan_bitwidth(model, n_bit):
    '''This script change the quantization bit-width of entire model to n_bit'''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            m.N_bits = n_bit
            # print("Change weight bit-width as {}.".format(m.N_bits))
            m.b_w.data = m.b_w.data[-m.N_bits:]
            m.b_w[0] = -m.b_w[0]
            print(m.b_w)
    return 
            

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

############# ACC #################
# class RecorderMeter(object):
#     """Computes and stores the minimum loss value and its epoch index"""
#
#     def __init__(self, total_epoch):
#         self.reset(total_epoch)
#
#     def reset(self, total_epoch):
#         assert total_epoch > 0
#         self.total_epoch = total_epoch
#         self.current_epoch = 0
#         self.epoch_losses = np.zeros((self.total_epoch, 2),
#                                      dtype=np.float32)  # [epoch, train/val]
#         self.epoch_losses = self.epoch_losses - 1
#
#         self.epoch_accuracy = np.zeros((self.total_epoch, 2),
#                                        dtype=np.float32)  # [epoch, train/val]
#         self.epoch_accuracy = self.epoch_accuracy
#
#     def update(self, idx, train_loss, train_acc, val_loss, val_acc):
#         assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
#             self.total_epoch, idx)
#         self.epoch_losses[idx, 0] = train_loss
#         self.epoch_losses[idx, 1] = val_loss
#         self.epoch_accuracy[idx, 0] = train_acc
#         self.epoch_accuracy[idx, 1] = val_acc
#         self.current_epoch = idx + 1
#         # return self.max_accuracy(False) == val_acc
#
#     def max_accuracy(self, istrain):
#         if self.current_epoch <= 0: return 0
#         if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
#         else: return self.epoch_accuracy[:self.current_epoch, 1].max()
#
#     def plot_curve(self, save_path):
#         title = 'the accuracy/loss curve of train/val'
#         dpi = 80
#         width, height = 1200, 800
#         legend_fontsize = 10
#         scale_distance = 48.8
#         figsize = width / float(dpi), height / float(dpi)
#
#         fig = plt.figure(figsize=figsize)
#         x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
#         y_axis = np.zeros(self.total_epoch)
#
#         plt.xlim(0, self.total_epoch)
#         plt.ylim(0, 100)
#         interval_y = 5
#         interval_x = 5
#         plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
#         plt.yticks(np.arange(0, 100 + interval_y, interval_y))
#         plt.grid()
#         plt.title(title, fontsize=20)
#         plt.xlabel('the training epoch', fontsize=16)
#         plt.ylabel('accuracy', fontsize=16)
#
#         y_axis[:] = self.epoch_accuracy[:, 0]
#         plt.plot(x_axis,
#                  y_axis,
#                  color='g',
#                  linestyle='-',
#                  label='train-accuracy',
#                  lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)
#
#         y_axis[:] = self.epoch_accuracy[:, 1]
#         plt.plot(x_axis,
#                  y_axis,
#                  color='y',
#                  linestyle='-',
#                  label='valid-accuracy',
#                  lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)
#
#         y_axis[:] = self.epoch_losses[:, 0]
#         plt.plot(x_axis,
#                  y_axis * 50,
#                  color='g',
#                  linestyle=':',
#                  label='train-loss-x50',
#                  lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)
#
#         y_axis[:] = self.epoch_losses[:, 1]
#         plt.plot(x_axis,
#                  y_axis * 50,
#                  color='y',
#                  linestyle=':',
#                  label='valid-loss-x50',
#                  lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)
#
#         if save_path is not None:
#             fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
#             print('---- save figure {} into {}'.format(title, save_path))
#         plt.close(fig)


############ MCC ######################
class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index, along with MCC"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_mcc = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_mcc = self.epoch_mcc

    def update(self, idx, train_loss, train_mcc, val_loss, val_mcc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_mcc[idx, 0] = train_mcc
        self.epoch_mcc[idx, 1] = val_mcc
        self.current_epoch = idx + 1

    def max_mcc(self, istrain):
        if self.current_epoch <= 0:
            return 0
        if istrain:
            return self.epoch_mcc[:self.current_epoch, 0].max()
        else:
            return self.epoch_mcc[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the MCC/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(-1, 1)  # MCC range is -1 to 1
        interval_y = 0.2
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(-1, 1 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('Training Epoch', fontsize=16)
        plt.ylabel('MCC', fontsize=16)

        y_axis[:] = self.epoch_mcc[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-MCC', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_mcc[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-MCC', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis * 50, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis * 50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)

def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(
        time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT,
                                       time.gmtime(time.time())))
    return string + '-{}'.format(random.randint(1, 10000))
