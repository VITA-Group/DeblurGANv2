import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from util.metrics import PSNR, SSIM
import pytorch_ssim
from util.pyssim import ssim
from PIL import Image

class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['A']
        inputs = img
        targets = data['B']
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        return inputs, targets

    def get_acc(self, output, target):

        psnr = PSNR(output, target)

        return psnr

    def get_loss(self, mean_loss, mean_psnr, output=None, target=None):
        return '{:.3f}; psnr={}'.format(mean_loss, mean_psnr)

    def visualize_data(self, writer, data, outputs, niter):
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        images = inv_normalize(vutils.make_grid(data['A']))
        writer.add_image('Images', images, niter)


def get_model(model_config):
    return DeblurModel()

