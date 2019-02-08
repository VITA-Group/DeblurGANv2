import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import logging

REPORT_EACH = 100


class MetricCounter():
    def __init__(self, exp_name):
        super(MetricCounter, self).__init__()
        self.writer = SummaryWriter(exp_name)
        logging.basicConfig(filename='{}.log'.format(exp_name), level=logging.DEBUG)
        self.clear()
        self.best_metric = 0

    def clear(self):
        self.G_loss = []
        self.D_loss = []
        self.content_loss = []
        self.adv_loss = []
        self.psnr = []
        self.ssim = []

    def add_losses(self, l_G, l_content, l_D=0):
        self.G_loss.append(l_G)
        self.content_loss.append(l_content)
        self.adv_loss.append(l_G - l_content)
        self.D_loss.append(l_D)

    def add_metrics(self, psnr, ssim):
        self.psnr.append(psnr)
        self.ssim.append(ssim)

    def loss_message(self):
        mean_loss = np.mean(self.G_loss[-REPORT_EACH:])
        mean_psnr = np.mean(self.psnr[-REPORT_EACH:])
        mean_ssim = np.mean(self.ssim[-REPORT_EACH:])
        return '{:.3f}; psnr={}; ssim={}'.format(mean_loss, mean_psnr, mean_ssim)

    def write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        self.writer.add_scalar('{}_G_Loss'.format(scalar_prefix), np.mean(self.G_loss), epoch_num)
        self.writer.add_scalar('{}_D_Loss'.format(scalar_prefix), np.mean(self.D_loss), epoch_num)
        self.writer.add_scalar('{}_G_Loss_content'.format(scalar_prefix), np.mean(self.content_loss), epoch_num)
        self.writer.add_scalar('{}_SSIM'.format(scalar_prefix), np.mean(self.ssim), epoch_num)
        self.writer.add_scalar('{}_PSNR'.format(scalar_prefix), np.mean(self.psnr), epoch_num)

    def update_best_model(self):
        cur_metric = np.mean(self.psnr)
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False


