#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Jiale Chen'
__mail__ = 'chenjlcv@mail.ustc.edu.cn'

import os
import sys
import gym
import argparse
import traceback
import numpy as np

import torch
import torch.nn as nn
from torch import multiprocessing as mp

from utils import *
from model import *
from optim import *
from test import test
from train import train

parser = argparse.ArgumentParser(description='A3C Compete')
# Basic parameters
parser.add_argument('--seed', default=123, type=int, help='seed random')
# A3C parameters
parser.add_argument('--num-processes', type=int, default=6, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=500000, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--batch-size-rl', type=int, default=16, metavar='SIZE', help='A3C batch size')
parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for a3c')
parser.add_argument('--lr-decay', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--param-init', type=str, default='xavier_normal',  help=('xavier_normal, xavier_uniform, ' 'kaiming_nornal, kaming_uniform'))
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=20000, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', default=False, help='Render evaluation agent')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
# Model training
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
# parser.add_argument('--lr', type=float, default=0.0007, metavar='η', help='Learning rate')
# parser.add_argument('--batch-size', type=int, default=64)
# parser.add_argument('--max-epoch', type=int, default=25)
# parser.add_argument('--env_lr', type=float, default=0.0001)
# parser.add_argument('--o_dir', type=str, default='/home/chenjl/Research/TF_GAN/weights', help='The path for outputs')

# parser.add_argument('--test_step', type=int, default=1000)
# parser.add_argument('--val_step', type=int, default=1000)
# parser.add_argument('--save_step', type=int, default=1000)
# parser.add_argument('--lmdb_train', type=str, default='/home/chenjl/Research/TF_GAN/datasets/', 
# 										help='The path for training datasets.')
# parser.add_argument('--lmdb_test', type=str, default='/home/chenjl/Research/TF_GAN/datasets/', 
# 										help='The path for testing datasets.')
# parser.add_argument('--lmdb_val', type=str, default='/home/chenjl/Research/TF_GAN/datasets/', 
# 										help='The path for testing datasets.')


# Training parameters
# parser.add_argument('--debug', action='store_true', default=False, help='The debug mode')
# parser.add_argument('--cuda', action='store_true', default=False, help='If use cuda, set it True.')

# Data argumentation settings
# parser.add_argument('--z_height', type=int, default=20, help='Resize input image for generator')
# parser.add_argument('--z_width', type=int, default=16,	help='Resize input image for generator')
# parser.add_argument('--img_height', type=int, default=128, help='Resize input image for discriminator')
# parser.add_argument('--img_width', type=int, default=128, help='Resize input image for discriminator')
# parser.add_argument('--c_dim', type=int, default=3, help='The channel of the images')

# RL argumentation
# parser.add_argument('--env', default='Pendulum-v0', type=str, help='gym environment')
# parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
# parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
# parser.add_argument('--test', default=False, type=bool, help='test mode sets lr=0, chooses most likely actions')
# parser.add_argument('--update_steps', default=20, type=int, help='steps to update')

# parser.add_argument('--gamma', default=0.9, type=float, help='discount for gamma-discounted rewards')
# parser.add_argument('--tau', default=1.0, type=float, help='discount for generalized advantage estimation')
# parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')

if __name__ == '__main__':
	# BLAS setup
	os.environ['OMP_NUM_THREADS'] = '1' # Not working on server
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['CUDA_VISIBLE_DEVICES'] = '5'

	# Setup
	args = parser.parse_args()
	print(' ' * 26 + 'Options')
	for k, v in vars(args).items():
		print(' ' * 26 + k + ': ' + str(v))

	args.env = 'CartPole-v1'
	args.save_dir = '{}/'.format(args.env.lower())
	os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None

	torch.manual_seed(args.seed)
	T = Counter()  # Global shared counter

	# Create shared network
	env = gym.make(args.env)
	shared_model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
	shared_model.share_memory()

	# Create optimiser for shared network parameters with shared statistics
	optimiser = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_decay)
	optimiser.share_memory()
	env.close()

	# # Start validation agent
	processes = []
	p = mp.Process(target=test, args=(0, args, T, shared_model))
	p.start()
	processes.append(p)

	if not args.evaluate:
		# Start training agents
		for rank in range(1, args.num_processes + 1):
			p = mp.Process(target=train, args=(rank, args, T, shared_model, optimiser))
			p.start()
			processes.append(p)

	# Clean up
	for p in processes:
		p.join()
