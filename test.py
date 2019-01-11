from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
from models.networks import get_nets
import cv2
import yaml
import os
from torchvision import models, transforms
from torch.autograd import Variable
import shutil
import glob
import tqdm
from util.metrics import PSNR
from albumentations import Compose, CenterCrop, PadIfNeeded
import random
from PIL import Image
from ssim import SSIM

def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--weights_path', required=True, help='pytorch weights path')
	parser.add_argument('--save_dir', required=True, help='path for image to test')

	return parser.parse_args()

def prepare_dirs(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)

def get_gt_image(path):
	dir, filename = os.path.split(path)
	base, _ = os.path.split(dir)
	img = cv2.cvtColor(cv2.imread(os.path.join(base, 'sharp', filename)), cv2.COLOR_BGR2RGB)
	return img

def test_image(model, save_path, image_path):

	img_transforms = transforms.Compose([
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	size_transform = Compose([
		PadIfNeeded(768, 1280)
	])
	crop = CenterCrop(720, 1280)
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_s = size_transform(image=img)['image']
	img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
	img_tensor = img_transforms(img_tensor)
	with torch.no_grad():
		img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
		result_image = model(img_tensor)
	result_image = result_image[0].cpu().float().numpy()
	result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
	result_image = result_image.astype('uint8')
	result_image = crop(image=result_image)['image']
	gt_image = get_gt_image(image_path)
	#gt_image = size_transform(image=gt_image)['image']
	_, filename = os.path.split(image_path)
	psnr = PSNR(result_image, gt_image)
	pilFake = Image.fromarray(result_image)
	pilReal = Image.fromarray(gt_image)
	ssim = SSIM(pilFake).cw_ssim_value(pilReal)
	#result_image = np.hstack((img, result_image, gt_image))
	#cv2.imwrite(os.path.join(save_path, filename), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
	return psnr, ssim

def test(model, args, files):
	psnr = 0
	ssim = 0
	for file in tqdm.tqdm(files):
		cur_psnr, cur_ssim = test_image(model, args.save_dir, file)
		psnr += cur_psnr
		ssim += cur_ssim
	print("PSNR = {}".format(psnr / len(files)))
	print("SSIM = {}".format(ssim / len(files)))


if __name__ == '__main__':
	args = get_args()
	with open('config/deblur_solver.yaml', 'r') as f:
		config = yaml.load(f)
	model, _ = get_nets(config['model'])
	model.load_state_dict(torch.load(args.weights_path)['model'])
	filenames = glob.glob(config['dataroot_val'] + '/test' + '/**/blur/*.png', recursive=True)
	#prepare_dirs(args.save_dir)
	test(model, args, filenames)
