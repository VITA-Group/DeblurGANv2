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


def estimate_blur(x):
	return cv2.Laplacian(x, cv2.CV_32F).var()


def get_gt_image(path):
	dir, filename = os.path.split(path)
	base, seq = os.path.split(dir)
	base, _ = os.path.split(base)
	img = cv2.cvtColor(cv2.imread(os.path.join(base, 'sharp', seq, filename)), cv2.COLOR_BGR2RGB)
	return img


def test_image(model, model2, model3, save_path, image_path):

	img_transforms = transforms.Compose([
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	crop = CenterCrop(704, 704)
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_s = crop(image=img)['image']
	img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
	img_tensor = img_transforms(img_tensor)
	with torch.no_grad():
		img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
		result_image = model(img_tensor)
		result_image2 = model2(img_tensor)
		result_image3 = model3(img_tensor)
	result_image = result_image[0].cpu().float().numpy()
	result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
	result_image = result_image.astype('uint8')
	result_image2 = result_image2[0].cpu().float().numpy()
	result_image2 = (np.transpose(result_image2, (1, 2, 0)) + 1) / 2.0 * 255.0
	result_image2 = result_image2.astype('uint8')
	result_image3 = result_image3[0].cpu().float().numpy()
	result_image3 = (np.transpose(result_image3, (1, 2, 0)) + 1) / 2.0 * 255.0
	result_image3 = result_image3.astype('uint8')
	gt_image = get_gt_image(image_path)
	gt_image = crop(image=gt_image)['image']
	lap = estimate_blur(result_image)
	lap_sharp = estimate_blur(gt_image)
	lap_blur = estimate_blur(img_s)
	_, filename = os.path.split(image_path)
	psnr = PSNR(result_image, gt_image)
	pilFake = Image.fromarray(result_image)
	pilReal = Image.fromarray(gt_image)
	ssim = SSIM(pilFake).cw_ssim_value(pilReal)
	result_image = np.hstack((img_s, result_image, result_image2, result_image3, gt_image))
	cv2.imwrite(os.path.join(save_path, filename), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
	return psnr, ssim, lap, lap_blur, lap_sharp

def test(model, model2, model3, args, files):
	psnr = 0
	ssim = 0
	lap = 0
	lap_sharp = 0
	lap_blur = 0
	for file in tqdm.tqdm(files):
		cur_psnr, cur_ssim, l, l_b, l_s = test_image(model, model2, model3, args.save_dir, file)
		psnr += cur_psnr
		ssim += cur_ssim
		lap += l
		lap_blur += l_b
		lap_sharp += l_s
	print("PSNR = {}".format(psnr / len(files)))
	print("SSIM = {}".format(ssim / len(files)))
	print("Var of Laplacian = {}".format(lap / len(files)))
	print("Var of Laplacian blurred = {}".format(lap_blur / len(files)))
	print("Var of Laplacian sharp = {}".format(lap_sharp / len(files)))


if __name__ == '__main__':
	args = get_args()
	with open('config/deblur_solver_test.yaml', 'r') as f:
		config = yaml.load(f)
	model, _ = get_nets(config['model'])
	config['model']['g_name'] = 'fpn_inception'
	model2, _ = get_nets(config['model'])
	config['model']['g_name'] = 'fpn_inception_simple'
	model3, _ = get_nets(config['model'])
	model.load_state_dict(torch.load(args.weights_path)['model'])
	model2.load_state_dict(torch.load('best_fpn.h5')['model'])
	model3.load_state_dict(torch.load('best_fpn_3840.h5')['model'])
	filenames = glob.glob(config['dataroot_val'] + '/test' + '/blur/**/*.png', recursive=True)
	filenames = random.sample(filenames, 50)
	prepare_dirs(args.save_dir)
	test(model, model2, model3, args, filenames)
