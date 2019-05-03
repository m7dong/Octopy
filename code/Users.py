import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from torch.autograd import Variable
from torch.utils import data
from warehouse.funcs import get_dataloader, save_checkpoint
import torch.nn.functional as F


class User(object):
	def __init__(self, user_index, ready_model, config):
		#print('U-1')
		self.update_local_config(config.users[user_index])
		self.user_index = user_index
		self.net = ready_model
		self.device = torch.device(next(self.net.parameters()).device)
		#print('U-2: ', type(self.device))
		#print('U-3')
		self.local_train_dataset, self.local_test_dataset = get_dataloader()  # just for testing
		self.local_train_loader = torch.utils.data.DataLoader(self.local_train_dataset, 
															batch_size=self.local_batchsize,
															shuffle=True)
		self.local_test_loader = torch.utils.data.DataLoader(self.local_test_dataset, 
															shuffle=False)

	def update_local_config(self, local_config):
		self.__dict__.update(local_config)

	def get_optimizer(self):
		if self.optimizer == 'SGD':
			return torch.optim.SGD(self.net.parameters(), lr=self.lr)
		return torch.optim.SGD(self.net.parameters(), lr=self.lr)

	def get_loss_func(self):
		if self.loss_func == 'nll':
			return F.nll_loss

	def local_train(self):
		print('Starting the training of user: ', self.user_index)
		optimizer = self.get_optimizer()
		loss_func = self.get_loss_func()
		self.net.train()
		for epoch in range(1, self.local_epoch + 1):
			#print('LOL, I am training...')
			for batch_idx, (data, target) in enumerate(self.local_train_loader):
				#print('U-3: ', target)
				data, target = data.to(self.device), target.to(self.device)
				#print('U-4: ', data)
				optimizer.zero_grad()
				#print('U-5: ', type(self.net))
				output = self.net(data)
				#print('U-6: ')
				loss = loss_func(output, target)
				#print('U-7: ')
				loss.backward()
				#print('U-8: ')
				optimizer.step()

		# save_checkpoint(self.net, filename='checkpoint_%d.pth' % self.user_index)		
		


