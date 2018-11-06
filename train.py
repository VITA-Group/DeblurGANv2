from __future__ import print_function

import torch
import torch.optim as optim
from data.data_loader import CreateDataLoader
import tqdm
import cv2
import yaml
from schedulers import WarmRestart
import numpy as np
from models.networks import get_net
from models.losses import get_loss
from models.models import get_model
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(filename='train.log',level=logging.DEBUG)
writer = SummaryWriter('runs')
REPORT_EACH = 8
torch.backends.cudnn.bencmark = True
cv2.setNumThreads(0)

class Trainer(object):
	def __init__(self, config):
		self.config = config
		self.train_dataset = self._get_dataset(config, config['datasets']['train'])
		self.val_dataset = self._get_dataset(config, config['datasets']['validation'])
		self.best_metric = 0
		self.warmup_epochs = config['warmup_num']


	def train(self):
		self._init_params()
		for epoch in range(0, config['num_epochs']):
			if epoch == self.warmup_epochs:
				self.netG.module.unfreeze()
				self.optimizer = self._get_optim()
				self.scheduler = self._get_scheduler()

			train_loss = self._run_epoch(epoch)
			val_loss, val_metric = self._validate(epoch)
			self.scheduler.step(val_loss)

			if val_metric > self.best_metric:
				self.best_metric = val_metric
				torch.save({
					'model': self.netG.state_dict()
				}, 'best_{}.h5'.format(self.config['experiment_desc']))
			torch.save({
				'model': self.netG.state_dict()
			}, 'last_{}.h5'.format(self.config['experiment_desc']))
			print(('val_loss={}, val_metric={}, best_metric={}\n'.format(val_loss, val_metric, self.best_metric)))
			logging.debug("Experiment Name: %s, Epoch: %d, Train Loss: %.3f, Val Accuracy: %.3f, Val Loss: %.3f, Best Loss: %.3f" % (
				self.config['experiment_desc'], epoch, train_loss, val_loss, val_metric, self.best_metric))

	def _run_epoch(self, epoch):
		losses = []
		accuracy = []
		batches_per_epoch = len(self.train_dataset) / config['batch_size']

		for param_group in self.optimizer.param_groups:
			lr = param_group['lr']
		tq = tqdm.tqdm(self.train_dataset)
		tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
		i = 0
		for data in tq:
			inputs, targets = self.model.get_input(data)
			self.optimizer.zero_grad()
			outputs = self.netG(inputs)
			loss = self.criterionG(outputs, targets)
			loss.backward()
			self.optimizer.step()
			losses.append(loss.data[0])
			accuracy.append(self.model.get_acc(outputs, targets))
			mean_loss = np.mean(losses[-REPORT_EACH:])
			mean_acc = np.mean(accuracy[-REPORT_EACH:])
			if i % 100 == 0:
				writer.add_scalar('Train_Loss', mean_loss, i + (batches_per_epoch * epoch))
				writer.add_scalar('Train_Metric', mean_acc, i + (batches_per_epoch * epoch))
				self.model.visualize_data(writer, data, outputs,  i + (batches_per_epoch * epoch))
			tq.set_postfix(loss=self.model.get_loss(mean_loss, mean_acc, outputs, targets))
			i += 1
		tq.close()
		return np.mean(losses)

	def _validate(self, epoch):
		losses = []
		accuracy = []
		tq = tqdm.tqdm(self.val_dataset)
		tq.set_description('Validation')
		for data in tq:
			inputs, targets = self.model.get_input(data)
			outputs = self.netG(inputs)
			loss = self.criterionG(outputs, targets)
			losses.append(loss.data[0])
			accuracy.append(self.model.get_acc(outputs, targets))
		val_loss = np.mean(losses)
		val_acc = np.mean(accuracy)
		tq.close()
		writer.add_scalar('Validation_Loss', val_loss, epoch)
		writer.add_scalar('Validation_Metric', val_acc, epoch)
		return val_loss, val_acc

	def _get_dataset(self, config, filename):
		data_loader = CreateDataLoader(config, filename)
		return data_loader.load_data()

	def _get_optim(self):
		if self.config['optimizer']['name'] == 'adam':
			optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=self.config['optimizer']['lr'])
		elif self.config['optimizer']['name'] == 'sgd':
			optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=self.config['optimizer']['lr'])
		elif self.config['optimizer']['name'] == 'adadelta':
			optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=self.config['optimizer']['lr'])
		else:
			raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
		return optimizer

	def _get_scheduler(self):
		if self.config['scheduler']['name'] == 'plateau':
			scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
															 mode='min',
															 patience=self.config['scheduler']['patience'],
															 factor=self.config['scheduler']['factor'],
															 min_lr=self.config['scheduler']['min_lr'])
		elif self.config['optimizer']['name'] == 'sgdr':
			scheduler = WarmRestart(self.optimizer)
		else:
			raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
		return scheduler

	def _init_params(self):
		self.netG, self.netD = get_net(self.config['model'])
		self.netG.cuda()
		self.netD.cuda()
		self.model = get_model(self.config['model'])
		self.criterionG, self.criterionD = get_loss(self.config['model'])
		self.optimizer = self._get_optim()
		self.scheduler = self._get_scheduler()


if __name__ == '__main__':
	with open('config/gan_solver.yaml', 'r') as f:
		config = yaml.load(f)
	trainer = Trainer(config)
	trainer.train()


