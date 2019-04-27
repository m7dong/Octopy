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
	def __init__(self, user_index, ready_model, learning_rate=0.0001, local_batchsize=100, local_epoch=10):
		self.user_index = user_index
		self.learning_rate = learning_rate
		self.local_batchsize = local_batchsize
		self.local_epoch = local_epoch
		self.net = ready_model
		self.device = next(self.net.parameters()).device

		self.local_train_loader, self.local_test_loader = get_dataloader()  # just for testing

	def local_train(self):
		model = self.net
		model.train()
		optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
		train_loader = self.local_train_loader

		for epoch in range(1, self.local_epoch + 1):
			for batch_idx, (data, target) in enumerate(train_loader):
				data, target = data.to(device), target.to(device)
				optimizer.zero_grad()
				output = model(data)
				loss = F.nll_loss(output, target)
				loss.backward()
				optimizer.step()
				
		


