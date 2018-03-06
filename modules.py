#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from collections import OrderedDict

class Conv2dAct(nn.Module):
	'''conv + activation
	'''

	def __init__(self, in_channels, out_channels, kernel_size, act_func, 
				stride=1, padding=0, padding_mode='zero', groups=1, 
				dilation=1, bias=True):
		super(Conv2dAct, self).__init__()
		self.body = nn.Sequential()

		if not isinstance(padding, int) or padding > 0:
			if padding_mode == 'zero':
				padding_layer = nn.ZeroPad2d
			elif padding_mode.startswith('rep'):
				padding_layer = nn.ReplicationPad2d
			elif padding_mode.startswith('ref'):
				padding_layer = nn.ReflectionPad2d
			self.body.add_module(
				'pad', padding_layer(padding)
			)

		self.body.add_module(
			'conv', torch.nn.Conv2d(in_channels=in_channels,
									out_channels=out_channels,
									kernel_size=kernel_size,
									stride=stride,
									padding=0,
									dilation=dilation,
									bias=bias,
									groups=groups))

		if act_func is not None:
			act_name = act_func.__class__.__name__
			self.body.add_module(act_name, act_func)

	def forward(self, x):
		return self.body(x)

class Conv2dBNAct(nn.Module):
	''' convolution with batch normalization
	'''

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, padding_mode='zero', dilation=1, groups=1,
				 bias=False, eps=1e-05, momentum=0.1, act_func=None):
		super(Conv2dBNAct, self).__init__()
		self.body = nn.Sequential()
		
		if not isinstance(padding, int) or padding > 0:
			if padding_mode == 'zero':
				padding_layer = nn.ZeroPad2d
			elif padding_mode.startswith('rep'):
				padding_layer = nn.ReplicationPad2d
			elif padding_mode.startswith('ref'):
				padding_layer = nn.ReflectionPad2d
			self.body.add_module(
				'pad', padding_layer(padding)
			)
		self.body.add_module(
			'conv', torch.nn.Conv2d(in_channels=in_channels,
									out_channels=out_channels,
									kernel_size=kernel_size,
									stride=stride,
									padding=0,
									dilation=dilation,
									groups=groups,
									bias=bias)
		)
		self.body.add_module(
			'bn', torch.nn.BatchNorm2d(in_channels,
									   eps=eps,
									   momentum=momentum)
		)

		if act_func is not None:
			act_name = act_func.__class__.__name__
			self.body.add_module(act_name, act_func)

	def forward(self, x):
		return self.body(x)

class ResidueBlock(torch.nn.Module):

	def __init__(self, in_channels, mid_channels, kernel_size,
				 act_func=None, padding_mode='zero', groups=1,
				 dilation=1, has_BN=True):
		super(ResidueBlock, self).__init__()
		if has_BN:
			key, conv2d, bias = 'bn_conv', Conv2dBNAct, False
		else:
			key, conv2d, bias = 'conv', Conv2dAct, True
		self.body = torch.nn.Sequential(OrderedDict([
			(key + '.1_1', conv2d(in_channels=in_channels,
								out_channels=mid_channels,
								kernel_size=kernel_size,
								bias=bias,
								act_func=act_func,
								stride=1,
								padding=kernel_size // 2,
								padding_mode=padding_mode,
								groups=groups,
								dilation=dilation)),
			(key + '.2', conv2d(in_channels=mid_channels,
								out_channels=mid_channels,
								kernel_size=kernel_size,
								bias=bias,
								act_func=None,
								stride=1,
								padding=kernel_size // 2,
								padding_mode=padding_mode,
								groups=groups,
								dilation=dilation))
		]))

	def forward(self, x):
		y = self.body(x)
		return x + y

class UpsampleBLock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, up_scale, 
				act_func, stride=1, padding=0, padding_mode='zero', groups=1, 
				dilation=1, bias=True):
		super(UpsampleBLock, self).__init__()
		self.body = nn.Sequential()
		
		if not isinstance(padding, int) or padding > 0:
			if padding_mode == 'zero':
				padding_layer = nn.ZeroPad2d
			elif padding_mode.startswith('rep'):
				padding_layer = nn.ReplicationPad2d
			elif padding_mode.startswith('ref'):
				padding_layer = nn.ReflectionPad2d
			self.body.add_module(
				'pad', padding_layer(padding)
			)

		self.body.add_module(
			'conv', torch.nn.Conv2d(in_channels=in_channels,
									out_channels=out_channels,
									kernel_size=kernel_size,
									stride=stride,
									padding=0,
									dilation=dilation,
									bias=bias,
									groups=groups)
		)

		self.body.add_module(
			'pixel_shuffle', torch.nn.PixelShuffle(2)
		)

		if act_func is not None:
			act_name = act_func.__class__.__name__
			self.body.add_module(act_name, act_func)

	def forward(self, x):
		return self.body(x)

class LinearAct(nn.Module):
	'''linear + activation
	'''

	def __init__(self, in_features, out_features, act_func, bias=True):
		super(LinearAct, self).__init__()
		self.body = nn.Sequential()

		self.body.add_module(
			'conv', torch.nn.Linear(in_features=in_features,
									out_features=out_features,
									bias=bias))

		if act_func is not None:
			act_name = act_func.__class__.__name__
			self.body.add_module(act_name, act_func)

	def forward(self, x):
		return self.body(x)
