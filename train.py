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
import logging
from models.networks import get_nets
from models.losses import get_loss
from models.models import get_model
from metric_counter import MetricCounter
cv2.setNumThreads(0)

class Trainer(object):
	def __init__(self, config):
		self.config = config
		self.train_dataset = self._get_dataset(config, 'train')
		self.val_dataset = self._get_dataset(config, 'test')
		self.adv_lambda = config['model']['adv_lambda']
		self.metric_counter = MetricCounter(config['experiment_desc'])
		self.warmup_epochs = config['warmup_num']

	def train(self):
		self._init_params()
		for epoch in range(0, config['num_epochs']):
			if (epoch == self.warmup_epochs) and not(self.warmup_epochs == 0):
				self.netG.module.unfreeze()
				self.optimizer_G = self._get_optim(self.netG)
				self.scheduler_G = self._get_scheduler(self.optimizer_G)
			self._run_epoch(epoch)
			self._validate(epoch)
			self.scheduler_G.step()
			self.scheduler_D.step()

			if self.metric_counter.update_best_model():
				torch.save({
					'model': self.netG.state_dict()
				}, 'best_{}.h5'.format(self.config['experiment_desc']))
			torch.save({
				'model': self.netG.state_dict()
			}, 'last_{}.h5'.format(self.config['experiment_desc']))
			print(self.metric_counter.loss_message())
			logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
				self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))

	def _run_epoch(self, epoch):
		self.metric_counter.clear()
		for param_group in self.optimizer_G.param_groups:
			lr = param_group['lr']
		tq = tqdm.tqdm(self.train_dataset.dataloader)
		tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
		i = 0
		for data in tq:
			inputs, targets = self.model.get_input(data)
			outputs = self.netG(inputs)
			self.optimizer_D.zero_grad()
			loss_D = self.adv_lambda * self.criterionD(self.netD, outputs, targets)
			loss_D.backward(retain_graph=True)
			self.optimizer_D.step()

			self.optimizer_G.zero_grad()
			loss_content = self.criterionG(outputs, targets)
			loss_adv = self.criterionD.get_g_loss(self.netD, outputs, targets)
			loss_G = loss_content + self.adv_lambda * loss_adv
			loss_G.backward()
			self.optimizer_G.step()
			self.metric_counter.add_losses(loss_G.item(), loss_D.item(), loss_content.item(), loss_adv.item())
			curr_psnr, curr_ssim = self.model.get_acc(outputs, targets)
			self.metric_counter.add_metrics(curr_psnr, curr_ssim)
			tq.set_postfix(loss=self.metric_counter.loss_message())
			i += 1
		tq.close()
		self.metric_counter.write_to_tensorboard(epoch)

	def _validate(self, epoch):
		self.metric_counter.clear()
		tq = tqdm.tqdm(self.val_dataset.dataloader)
		tq.set_description('Validation')
		for data in tq:
			inputs, targets = self.model.get_input(data)
			outputs = self.netG(inputs)
			loss_D = self.adv_lambda * self.criterionD(self.netD, outputs, targets)
			loss_content = self.criterionG(outputs, targets)
			loss_adv = self.criterionD.get_g_loss(self.netD, outputs, targets)
			loss_G = loss_content + self.adv_lambda * loss_adv
			self.metric_counter.add_losses(loss_G.item(), loss_D.item(), loss_content.item(), loss_adv.item())
			curr_psnr, curr_ssim = self.model.get_acc(outputs, targets, full=True)
			self.metric_counter.add_metrics(curr_psnr, curr_ssim)
		tq.close()
		self.metric_counter.write_to_tensorboard(epoch, validation=True)

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
	with open('config/deblur_solver.yaml', 'r') as f:
		config = yaml.load(f)
	trainer = Trainer(config)
	trainer.train()


