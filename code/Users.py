import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.autograd import Variable
from torch.utils import data
from warehouse.funcs import get_dataloader, save_checkpoint
import torch.nn.functional as F


class User(object):
	def __init__(self, user_index, ready_model, learning_rate=0.0001, 
					loss_func=None, local_batchsize=100, local_epoch=10, optimizer='SGD'):
		#print('U-1')
		self.user_index = user_index
		self.net = ready_model
		self.device = torch.device(next(self.net.parameters()).device)
		#print('U-2: ', type(self.device))
		self.learning_rate = learning_rate
		self.local_batchsize = local_batchsize
		self.local_epoch = local_epoch
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
		self.loss_func = F.nll_loss
		#print('U-3')
		self.local_train_dataset, self.local_test_dataset = get_dataloader()  # just for testing
		self.local_train_loader = torch.utils.data.DataLoader(self.local_train_dataset, 
															batch_size=self.local_batchsize,
															shuffle=True)
		self.local_test_loader = torch.utils.data.DataLoader(self.local_test_dataset, 
															shuffle=False)


	def local_train(self):
		print('Starting the training of user: ', self.user_index)
		self.net.train()
		for epoch in range(1, self.local_epoch + 1):
			#print('LOL, I am training...')
			for batch_idx, (data, target) in enumerate(self.local_train_loader):
				#print('U-3: ', target)
				data, target = data.to(self.device), target.to(self.device)
				#print('U-4: ', data)
				self.optimizer.zero_grad()
				#print('U-5: ', type(self.net))
				output = self.net(data)
				#print('U-6: ')
				loss = self.loss_func(output, target)
				#print('U-7: ')
				loss.backward()
				#print('U-8: ')
				self.optimizer.step()

		save_checkpoint(self.net, filename='checkpoint_%d.pth' % user_index)		
		


