import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
from util.metrics import PSNR
from skimage.measure import compare_ssim as SSIM
from PIL import Image
import cv2
import os


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['A']
        inputs = img
        targets = data['B']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_acc(self, output, target):
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)

        return psnr, ssim


def get_model(model_config):
    return DeblurModel()

