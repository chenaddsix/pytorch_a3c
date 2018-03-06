#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Jiale Chen'
__mail__ = 'chenjlcv@mail.ustc.edu.cn'

import os
import gym
import torch
import argparse
import numpy as np

import utils
from utils import *
from dataset import *
from model import SRGAN_g, VGGNet
from tensorboardX import SummaryWriter

from torch.autograd import Variable
import torch.nn.functional as F

class BasicTask:
	def __init__(self):
		self.normalized_state = False

	def normalize_state(self, state):
		return state

	def reset(self):
		state = self.env.reset()
		if self.normalized_state:
			return self.normalize_state(state)
		return state

	def step(self, action):
		next_state, reward, done, info = self.env.step(action)
		if self.normalized_state:
			next_state = self.normalize_state(next_state)
		return next_state, np.sign(reward), done, info

	def random_action(self):
		return self.env.action_space.sample()

class Pendulum(BasicTask):
	name = 'Pendulum-v0'
	success_threshold = -10

	def __init__(self):
		BasicTask.__init__(self)
		self.env = gym.make(self.name)
		self.max_episode_steps = self.env._max_episode_steps
		self.env._max_episode_steps = 200
		self.action_dim = self.env.action_space.shape[0]
		self.state_dim = self.env.observation_space.shape[0]

	def step(self, action):
		action = np.clip(action, -2, 2)
		next_state, reward, done, info = self.env.step(action)
		return next_state, reward, done, info

class FaceRestore_v0(BasicTask):
	def __init__(self, 
				 args, 
				 global_net, 
				 vgg_net, 
				 train_dataloader, 
				 test_dataloader):
		BasicTask.__init__(self)
		self.args = args
		# Datasets
		self.train_dataloader = train_dataloader
		self.test_dataloader = test_dataloader

		# Network
		self.global_net = global_net
		self.net_g = SRGAN_g()
		self.vgg_net = vgg_net
		self.net_g.load_state_dict(self.global_net.state_dict())

		if args.cuda:
			self.global_net = self.global_net.cuda()
			self.net_g = self.net_g.cuda()
			self.vgg_net = self.vgg_net.cuda()

		# Optimizer
		self.opt = torch.optim.Adam(self.net_g.parameters(),
							   lr=args.lr,
							   weight_decay=1e-6)
		self.global_opt = torch.optim.Adam(self.global_net.parameters(),
							   lr=args.lr,
							   weight_decay=1e-6)

		# loss functions
		self.loss_items = {
			'vgg': {'func': utils.ContentLoss(self.vgg_net), 'factor': 2e-6},
			'mse': {'func': F.mse_loss, 'factor': 1.0}
		}

		# Summary
		self.writer = SummaryWriter(args.o_dir)

		# if args.resume:
		# 	self.net_g.load_state_dict(torch.load('../weights/SR_epoch09.pth'))

		# RL param
		self.max_episode_steps = 100
		self.action_dim = 1
		self.state_dim = 1
		self.env_step = 0

	def normalize_state(self, state):
		return state

	def scale_reward(self, reward):
		return reward

	def adjust_learning_rate(self, learning_rate):
	    for param_group in self.opt.param_groups:
	        param_group['lr'] = learning_rate

	def compute_loss(self, network_outputs, targets):
		loss_value = 0.0
		for loss_name in self.loss_items:
			v = self.loss_items[loss_name]['func'](network_outputs, targets)
			loss_value += v * self.loss_items[loss_name]['factor']
		return loss_value

	def get_data(self, requires_grad=False):
		'''
		Get the downsample images and orignal images for training and testing
		'''
		inputs, targets = batch2feeds(self.train_dataloader.next())
		if self.args.cuda:
			inputs, targets = Variable(inputs.cuda(), requires_grad=requires_grad), Variable(targets.cuda())
		else:
			inputs, targets = Variable(inputs, requires_grad=requires_grad), Variable(targets)
		return inputs, targets

	def get_state(self, inputs, net_g_outputs, targets):
		'''
		Get the observation. Observation is the mean of a batch of images. 
		'''
		self.net_g.zero_grad()
		net_g_loss = self.compute_loss(net_g_outputs, targets)
		net_g_loss.backward()

		state = inputs.grad.data.mean(dim=0)
		state = self.normalize_state(state) # preprocess the observation
		return state

	def network_evaluate(self, args, net_g_outputs, global_outputs, targets, inputs):
		net_g_outputs = np.transpose(net_g_outputs.data.cpu().numpy(), (0,2,3,1))
		global_outputs = np.transpose(global_outputs.data.cpu().numpy(), (0,2,3,1))
		targets = np.transpose(targets.data.cpu().numpy(), (0,2,3,1))
		global_img = (global_outputs+1.)*127.5; net_g_img = (net_g_outputs+1.)*127.5; targets_img = (targets+1.)*127.5
		net_g_psnr = get_imgs_psnr(net_g_img, targets_img)
		global_psnr = get_imgs_psnr(global_img, targets_img)

		# Save image
		if args.info['steps'][0] % 5000 == 0:
			num_steps = int(self.args.info['steps'][0])
			inputs = np.transpose(inputs.data.cpu().numpy(), (0,2,3,1))
			inputs = (inputs+1.)*127.5
			save_imgs(net_g_img, global_img, targets_img, inputs, './net_g_'+str(num_steps)+'.png')
			torch.save(self.net_g.state_dict(), save_dir+'/neg_g_{:.0f}.pth'.format(num_steps))
			torch.save(self.global_net.state_dict(), save_dir+'/global_{:.0f}.pth'.format(num_steps))

		return net_g_psnr, global_psnr

	def reset(self, args):
		self.env_step = 0
		print('====================== Reset: {} ===================='.format(int(args.info['steps'][0])))
		# Evaluate the network
		inputs, targets = self.get_data()
		net_g_outputs = self.net_g(inputs)
		global_outputs = self.global_net(inputs)
		net_g_psnr, global_psnr = self.network_evaluate(args, net_g_outputs, global_outputs, targets, inputs)
		print('Net_g PSNR: {}, Global PSNR:{}'.format(net_g_psnr, global_psnr))

		# Reset the network weights
		if net_g_psnr > global_psnr:
			print('-------------------Local net is better than global net!------------------')
			self.global_net.load_state_dict(self.net_g.state_dict())
		else:
			print('-------------------Global net is better than local net!------------------')
			self.net_g.load_state_dict(self.global_net.state_dict())

		# Set train mode
		self.net_g.train()
		# Get the training data
		inputs, targets = self.get_data(requires_grad=True)
		net_g_outputs = self.net_g(inputs)

		# Compute Loss and Gradient
		state = self.get_state(inputs, net_g_outputs, targets)

		return state

	def step(self, args, action):
		info = None
		self.env_step += 1
		# Perform the action
		self.adjust_learning_rate(action)
		
		# get the local and global outputs
		inputs, targets = self.get_data()
		net_g_outputs = self.net_g(inputs)
		global_outputs = self.global_net(inputs)

		# Compute the loss and update the network
		self.net_g.zero_grad(); self.global_net.zero_grad()
		net_g_loss = self.compute_loss(net_g_outputs, targets)
		global_loss = self.compute_loss(global_outputs, targets)
		net_g_loss.backward()
		global_loss.backward()

		# Logging
		num_steps = int(args.info['steps'][0])
		self.writer.add_scalar('net_g_loss', net_g_loss, num_steps)
		self.writer.add_scalar('global_loss', global_loss, num_steps)

		self.opt.step()
		self.global_opt.step()
		
		# Evaluate the acton
		inputs, targets = self.get_data(requires_grad=True)
		net_g_outputs = self.net_g(inputs)
		global_outputs = self.global_net(inputs)

		state = self.get_state(inputs, net_g_outputs, targets) # Get the starte to reture

		net_g_psnr, global_psnr = self.network_evaluate(args, net_g_outputs, global_outputs, targets, inputs)

		reward = np.mean(net_g_psnr) - np.mean(global_psnr) # the reward for the action

		terminal = False or self.env_step >= self.max_episode_steps

		return state, reward, terminal, info

