import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.autograd import Variable
from torch.utils import data
from funcs import get_dataloader
import torch.nn.functional as F


class User(object):
	def __init__(self, user_index, ready_model, learning_rate=0.0001, 
					loss_func=None, local_batchsize=100, local_epoch=10, optimizer='SGD'):
		self.user_index = user_index
		self.net = ready_model
		self.device = next(self.net.parameters()).device
		
		self.learning_rate = learning_rate
		self.local_batchsize = local_batchsize
		self.local_epoch = local_epoch
		self.optimizer = self.set_optimizer(optimizer)
		self.loss_func = self.set_loss_func(loss_func)

		self.local_train_loader, self.local_test_loader = get_dataloader()  # just for testing


	def set_loss_func(self, loss_func):
		return F.nll_loss


	def set_optimizer(self, optimizer):
		if self.optimizer = 'Adam':
			return torch.optim.Adam

		return torch.optim.SGD

	def local_train(self):
		model = self.net
		model.train()
		optimizer = self.optimizer(net.parameters(), lr=self.learning_rate)
		train_loader = self.local_train_loader

		for epoch in range(1, self.local_epoch + 1):
			for batch_idx, (data, target) in enumerate(train_loader):
				data, target = data.to(device), target.to(device)
				optimizer.zero_grad()
				output = model(data)
				loss = self.loss_func(output, target)
				loss.backward()
				optimizer.step()
				
		


