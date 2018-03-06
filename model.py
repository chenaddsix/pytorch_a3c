#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

import pyro
import numpy as np
from modules import *
import pyro.distributions as dist

class SRGAN_g(nn.Module):
	'''SRGAN_g
	'''
	def __init__(self):
		super(SRGAN_g, self).__init__()
		
		self.feature_1 = nn.Sequential()
		self.residual = nn.Sequential()
		self.feature_2 = nn.Sequential()

		self.feature_1.add_module(
			'conv.1', Conv2dAct(in_channels=3,
								out_channels=64,
								kernel_size=9,
								stride=1,
								padding=4, 
								act_func=nn.PReLU(), 
								bias=True))

		for i in range(16):
			self.residual.add_module(
				'res.%s' % i, ResidueBlock(in_channels=64, 
										mid_channels=64, 
										kernel_size=3,
										act_func=nn.PReLU(), 
										has_BN=True))


		self.residual.add_module(
			'res_conv', Conv2dBNAct(in_channels=64, 
								  out_channels=64, 
								  kernel_size=3, 
								  stride=1, 
								  padding=1, 
								  bias=False,
								  act_func=None))

		self.feature_2.add_module(
			'upsample.1', UpsampleBLock(in_channels=64, 
										out_channels=256, 
										kernel_size=3, 
										up_scale=2, 
										act_func=nn.PReLU(), 
										stride=1, 
										padding=1, 
										bias=True))
		self.feature_2.add_module(
			'upsample.2', UpsampleBLock(in_channels=64, 
										out_channels=256, 
										kernel_size=3, 
										up_scale=2, 
										act_func=nn.PReLU(), 
										stride=1, 
										padding=1, 
										bias=True))
		self.feature_2.add_module(
			'upsample.3', UpsampleBLock(in_channels=64, 
										out_channels=256, 
										kernel_size=3, 
										up_scale=2, 
										act_func=nn.PReLU(), 
										stride=1, 
										padding=1, 
										bias=True))
		self.feature_2.add_module(
			'conv.2', Conv2dAct(in_channels=64,
								out_channels=3,
								kernel_size=9,
								stride=1,
								padding=4, 
								act_func=nn.Tanh(), 
								bias=True))

	def forward(self, X):
		n = self.feature_1(X)
		temp = n
		n = self.residual(n) + temp
		n = self.feature_2(n)
		return n

class VGGNet(nn.Module):
	'''extract high-level features for content loss
	'''

	def __init__(self, weights=None, k=21):
		'''use layers before the kth layer
		'''
		super(VGGNet, self).__init__()
		if weights is None:
			net = models.vgg16_bn(pretrained=True)
		else:
			net = models.vgg16_bn()
			net.load_state_dict(torch.load(weights))
		self.net = nn.Sequential(*list(net.features.children())[:k])
		for p in self.net.parameters():
			p.requires_grad = False

	def forward(self, x):
		return self.net(x)

class ActorNet(nn.Module):
	'''Actor network
	'''
	def __init__(self):
		super(ActorNet, self).__init__()
		self.net = nn.Sequential()

		self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
		self.linear1 = nn.Linear(10240, 256)

		self.action_mean = nn.Linear(256, 1)

		self.action_std = nn.Linear(256, 1)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x))
		x = F.leaky_relu(self.conv2(x))
		x = F.leaky_relu(self.conv3(x))
		x = F.leaky_relu(self.conv4(x))
		x = x.view(x.size(0), -1)
		x = F.leaky_relu(self.linear1(x))
		mean = 5e-4 * F.sigmoid(self.action_mean(x))
		std = F.softplus(self.action_std(x))

		action = dist.normal(mean, std)
		# remove it from the graph
		action = action.detach()
		# calculate the log probability
		log_p = dist.normal.log_pdf(action, mean, std)

		entropy = self.entropy(std)

		return action, log_p, entropy

	def entropy(self, std):
		'''The entropy of the Gussian distribution
		'''
		return 0.5 * (1 + (2 * std.pow(2) * np.pi + 1e-5).log()).sum(1).mean()

class CriticNet(nn.Module):
	'''Critic network
	'''
	def __init__(self):
		super(CriticNet, self).__init__()

		self.net = nn.Sequential()

		self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
		self.linear1 = nn.Linear(10240, 256)

		self.linear2 = nn.Linear(256, 1)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x))
		x = F.leaky_relu(self.conv2(x))
		x = F.leaky_relu(self.conv3(x))
		x = F.leaky_relu(self.conv4(x))
		x = x.view(x.size(0), -1)
		x = F.leaky_relu(self.linear1(x))
		v = F.softplus(self.linear2(x))
		return v

class ActorCritic(nn.Module):
	def __init__(self, observation_space, action_space, hidden_size):
		super(ActorCritic, self).__init__()
		self.state_size = observation_space.shape[0]
		self.action_size = action_space.n

		self.relu = nn.ReLU(inplace=True)
		self.softmax = nn.Softmax(dim=1)

		self.fc1 = nn.Linear(self.state_size, hidden_size)
		self.lstm = nn.LSTMCell(hidden_size, hidden_size)
		self.fc_actor = nn.Linear(hidden_size, self.action_size)
		self.fc_critic = nn.Linear(hidden_size, 1)

	def forward(self, x, h):
		x = self.relu(self.fc1(x))
		h = self.lstm(x, h)  # h is (hidden state, cell state)
		x = h[0]
		policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
		V = self.fc_critic(x)
		return policy, V, h

