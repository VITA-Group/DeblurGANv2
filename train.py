from __future__ import print_function
import shutil
import torch
import torch.optim as optim
from data.data_loader import CreateDataLoader
import tqdm
import cv2
import os
import yaml
from schedulers import WarmRestart, LinearDecay
import numpy as np
from models.networks import get_nets
from models.losses import get_loss
from models.models import get_model
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(filename='res.log',level=logging.DEBUG)
writer = SummaryWriter('fpn_se')
REPORT_EACH = 100
torch.backends.cudnn.bencmark = True
cv2.setNumThreads(0)

class Trainer(object):
	def __init__(self, config):
		self.config = config
		self.train_dataset = self._get_dataset(config, 'train')
		self.val_dataset = self._get_dataset(config, 'test')
		self.best_metric = 0
		self.warmup_epochs = config['warmup_num']


	def train(self):
		self._init_params()
		for epoch in range(0, config['num_epochs']):
			if (epoch == self.warmup_epochs) and not(self.warmup_epochs == 0):
				self.netG.module.unfreeze()
				self.optimizer_G = self._get_optim(self.netG)
				self.scheduler_G = self._get_scheduler(self.optimizer_G)

			train_loss = self._run_epoch(epoch)
			val_loss, val_psnr = self._validate(epoch)
			self.scheduler_G.step()

			val_metric = val_psnr

			if val_metric > self.best_metric:
				self.best_metric = val_metric
				torch.save({
					'model': self.netG.state_dict()
				}, 'best_{}.h5'.format(self.config['experiment_desc']))
			print(('val_loss={}, val_metric={}, best_metric={}\n'.format(val_loss, val_metric, self.best_metric)))
			logging.debug("Experiment Name: %s, Epoch: %d, Train Loss: %.3f, Val Accuracy: %.3f, Val Loss: %.3f, Best Loss: %.3f" % (
				self.config['experiment_desc'], epoch, train_loss, val_loss, val_metric, self.best_metric))

	def _run_epoch(self, epoch):
		losses_G = []
		losses_vgg = []
		losses_adv = []
		psnrs = []
		ssim = []
		batches_per_epoch = len(self.train_dataset) / config['batch_size']

		for param_group in self.optimizer_G.param_groups:
			lr = param_group['lr']
		tq = tqdm.tqdm(self.train_dataset.dataloader)
		tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
		i = 0
		for data in tq:
			inputs, targets = self.model.get_input(data)
			outputs = self.netG(inputs)
			self.optimizer_D.zero_grad()
			loss_D = 0.001 * self.criterionD(self.netD, outputs, targets)
			loss_D.backward(retain_graph=True)
			self.optimizer_D.step()

			self.optimizer_G.zero_grad()
			loss_content = self.criterionG(outputs, targets)
			loss_adv = self.criterionD.get_g_loss(self.netD, outputs, targets)
			loss_G = loss_content + 0.001 * loss_adv
			loss_G.backward()
			self.optimizer_G.step()
			losses_G.append(loss_G.item())
			losses_vgg.append(loss_content.item())
			losses_adv.append(loss_adv.item())
			curr_psnr, curr_ssim = self.model.get_acc(outputs, targets)
			psnrs.append(curr_psnr)
			ssim.append(curr_ssim)
			mean_loss_G = np.mean(losses_G[-REPORT_EACH:])
			mean_loss_vgg = np.mean(losses_vgg[-REPORT_EACH:])
			mean_loss_adv = np.mean(losses_adv[-REPORT_EACH:])
			mean_psnr = np.mean(psnrs[-REPORT_EACH:])
			mean_ssim = np.mean(ssim[-REPORT_EACH:])
			if i % 1000 == 0:
				writer.add_scalar('Train_G_Loss', mean_loss_G, i + (batches_per_epoch * epoch))
				writer.add_scalar('Train_G_Loss_vgg', mean_loss_vgg, i + (batches_per_epoch * epoch))
				writer.add_scalar('Train_G_Loss_adv', mean_loss_adv, i + (batches_per_epoch * epoch))
				writer.add_scalar('Train_PSNR', mean_psnr, i + (batches_per_epoch * epoch))
				writer.add_scalar('Train_SSIM', mean_ssim, i + (batches_per_epoch * epoch))
			tq.set_postfix(loss=self.model.get_loss(mean_loss_G, mean_psnr, mean_ssim, outputs, targets))
			i += 1
		tq.close()
		return np.mean(losses_G)

	def _validate(self, epoch):
		losses = []
		psnrs = []
		ssim = []
		tq = tqdm.tqdm(self.val_dataset.dataloader)
		tq.set_description('Validation')
		for data in tq:
			inputs, targets = self.model.get_input(data)
			outputs = self.netG(inputs)
			loss_content = self.criterionG(outputs, targets)
			loss_G = loss_content + 0.001 * self.criterionD.get_g_loss(self.netD, outputs)
			losses.append(loss_G.item())
			curr_psnr, curr_ssim = self.model.get_acc(outputs, targets, full=True)
			psnrs.append(curr_psnr)
			ssim.append(curr_ssim)
		val_loss = np.mean(losses)
		val_psnr = np.mean(psnrs)
		val_ssim = np.mean(ssim)
		tq.close()
		writer.add_scalar('Validation_Loss', val_loss, epoch)
		writer.add_scalar('Validation_PSNR', val_psnr, epoch)
		writer.add_scalar('Validation_SSIM', val_ssim, epoch)
		return val_loss, val_psnr

	def _get_dataset(self, config, filename):
		data_loader = CreateDataLoader(config, filename)
		return data_loader.load_data()

	def _get_optim(self, model, disc=False):
		lr_multiplier = 2.0 if disc is True else 1.0
		if self.config['optimizer']['name'] == 'adam':
			optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config['optimizer']['lr'] * lr_multiplier)
		elif self.config['optimizer']['name'] == 'sgd':
			optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config['optimizer']['lr'] * lr_multiplier)
		elif self.config['optimizer']['name'] == 'adadelta':
			optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config['optimizer']['lr'] * lr_multiplier)
		else:
			raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
		return optimizer

	def _get_scheduler(self, optimizer):
		if self.config['scheduler']['name'] == 'plateau':
			scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
															 mode='min',
															 patience=self.config['scheduler']['patience'],
															 factor=self.config['scheduler']['factor'],
															 min_lr=self.config['scheduler']['min_lr'])
		elif self.config['optimizer']['name'] == 'sgdr':
			scheduler = WarmRestart(optimizer)
		elif self.config['scheduler']['name'] == 'linear':
			scheduler = LinearDecay(optimizer,
									min_lr=self.config['scheduler']['min_lr'],
									num_epochs=self.config['num_epochs'],
									start_epoch=self.config['scheduler']['start_epoch'])
		else:
			raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
		return scheduler

	def _init_params(self):
		self.netG, self.netD = get_nets(self.config['model'])
		self.netG.cuda()
		self.netD.cuda()
		self.model = get_model(self.config['model'])
		self.criterionG, self.criterionD = get_loss(self.config['model'])
		self.optimizer_G = self._get_optim(self.netG)
		self.optimizer_D = self._get_optim(self.netD)
		self.scheduler_G = self._get_scheduler(self.optimizer_G)
		self.scheduler_D = self._get_scheduler(self.optimizer_D)


if __name__ == '__main__':
	if os.path.exists('train_images'):
		shutil.rmtree('train_images')
	os.makedirs('train_images')
	with open('config/deblur_solver.yaml', 'r') as f:
		config = yaml.load(f)
	trainer = Trainer(config)
	trainer.train()


