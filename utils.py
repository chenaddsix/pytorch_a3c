#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp

import math
import plotly
from plotly.graph_objs import Scatter, Line

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from scipy.ndimage import gaussian_filter
from scipy import misc

def get_imgs_psnr(imgs1, imgs2):
	# per image
	def get_psnr(img1, img2):
		mse = np.mean((img1 - img2) ** 2)
		if mse == 0:
			return 100
		pixel_max = 255.0
		# return 10 * math.log10((pixel_max ** 2) / mse)
		return 20 * math.log10(pixel_max / math.sqrt(mse))
	assert imgs1.shape[0] == imgs2.shape[0], "Batch size is not match."
	assert np.mean(imgs1) >= 1.0, "Check 1st input images range"
	assert np.mean(imgs2) >= 1.0, "Check 2nd input images range"
	PSNR_RGB = list()

	for num_of_batch in range(imgs1.shape[0]):
		PSNR_RGB.append(get_psnr(imgs1[num_of_batch], imgs2[num_of_batch]))

	PSNR_RGB = np.mean(np.array(PSNR_RGB))

	return PSNR_RGB

# imgs1/imgs2: 0 ~ 255, float32, numpy, rgb
# [batch_size, height, width, channel]
# return: list
def get_imgs_psnr_ssim(imgs1, imgs2):
	# per image
	def get_psnr(img1, img2):
		mse = np.mean((img1 - img2) ** 2)
		if mse == 0:
			return 100
		pixel_max = 255.0
		# return 10 * math.log10((pixel_max ** 2) / mse)
		return 20 * math.log10(pixel_max / math.sqrt(mse))

	def block_view(A, block=(3, 3)):
		"""Provide a 2D block view to 2D array. No error checking made.
		Therefore meaningful (as implemented) only for blocks strictly
		compatible with the shape of A."""
		# simple shape and strides computations may seem at first strange
		# unless one is able to recognize the 'tuple additions' involved ;-)
		shape = (A.shape[0] / block[0], A.shape[1] / block[1]) + block
		strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
		return ast(A, shape=shape, strides=strides)

	def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
		bimg1 = block_view(img1, (4, 4))
		bimg2 = block_view(img2, (4, 4))
		s1 = np.sum(bimg1, (-1, -2))
		s2 = np.sum(bimg2, (-1, -2))
		ss = np.sum(bimg1*bimg1, (-1, -2)) + np.sum(bimg2*bimg2, (-1, -2))
		s12 = np.sum(bimg1*bimg2, (-1, -2))
		vari = ss - s1*s1 - s2*s2
		covar = s12 - s1*s2
		ssim_map = (2 * s1 * s2 + C1) * (2 * covar + C2) / ((s1 * s1 + s2 * s2 + C1) * (vari + C2))
		return np.mean(ssim_map)

	# FIXME there seems to be a problem with this code
	def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
		mu1 = gaussian_filter(img1, sd)
		mu2 = gaussian_filter(img2, sd)
		mu1_sq = mu1 * mu1
		mu2_sq = mu2 * mu2
		mu1_mu2 = mu1 * mu2
		sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
		sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
		sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2
		ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
		ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
		ssim_map = ssim_num / ssim_den
		return np.mean(ssim_map)
	assert imgs1.shape[0] == imgs2.shape[0], "Batch size is not match."
	assert np.mean(imgs1) >= 1.0, "Check 1st input images range"
	assert np.mean(imgs2) >= 1.0, "Check 2nd input images range"
	PSNR_RGB, SSIM = list(), list()
	for num_of_batch in range(imgs1.shape[0]):
		# RGB
		PSNR_RGB.append(get_psnr(imgs1[num_of_batch], imgs2[num_of_batch]))
		SSIM.append(ssim_exact(imgs1[num_of_batch], imgs2[num_of_batch]))
	PSNR_RGB = np.mean(np.array(PSNR_RGB))
	SSIM = np.mean(np.array(SSIM))
	return PSNR_RGB, SSIM

def save_imgs(img, global_img, origin_img, downsample_img, file_name):
	'''Save image
	'''
	N_samples, height, width, channel = img.shape
	N_row = int(N_samples/4)
	N_col = 4*4

	repeat_ds_img = []
	for index in range(N_samples):
		#print downsample_img.shape
		repeat_ds_img.append(downsample_img[index].repeat(axis=0, repeats=8).repeat(axis=1, repeats=8).astype(np.uint8))
	repeat_ds_img = np.array(repeat_ds_img)

	combined_imgs = np.ones((N_row*height, N_col*width, channel))
	for i in range(N_row):
		for j in range(int(N_col/4)):
			n = 4*j
			m = 4*j+1
			l = 4*j+2
			k = 4*j+3
			combined_imgs[int(i*height):int((i+1)*height), int(n*width):int((n+1)*width), :] = repeat_ds_img[int(i*(N_col/4)+j)]
			combined_imgs[int(i*height):int((i+1)*height), int(m*width):int((m+1)*width), :] = origin_img[int(i*(N_col/4)+j)]
			combined_imgs[int(i*height):int((i+1)*height), int(l*width):int((l+1)*width), :] = img[int(i*(N_col/4)+j)]
			combined_imgs[int(i*height):int((i+1)*height), int(k*width):int((k+1)*width), :] = global_img[int(i*(N_col/4)+j)]
	print('Saving the images.', file_name)
	misc.imsave(file_name, combined_imgs)

# Converts a state from the OpenAI Gym (a numpy array) to a batch tensor
def state_to_tensor(state):
	return torch.from_numpy(state).float().unsqueeze(0)

class ParamInit(object):
	'''parameter initializer
	'''

	def __init__(self, method, **kargs):
		super(ParamInit, self).__init__()
		self.inits = {
			'xavier_normal': torch.nn.init.xavier_normal,
			'xavier_uniform': torch.nn.init.xavier_uniform,
			'kaming_normal': torch.nn.init.kaiming_normal,
			'kaming_uniform': torch.nn.init.kaiming_uniform,
		}
		if method not in self.inits:
			raise RuntimeError('unknown initialization %s' % method)
		self.method = self.inits[method]
		self.kargs = kargs

	def __call__(self, module):
		classname = module.__class__.__name__
		if 'Conv' in classname and hasattr(module, 'weight'):
			self.method(module.weight, **self.kargs)
		elif 'BatchNorm' in classname:
			module.weight.data.normal_(1.0, 0.02)
			module.bias.data.fill_(0)

# Global counter
class Counter():
	def __init__(self):
		self.val = mp.Value('i', 0)
		self.lock = mp.Lock()

	def increment(self):
		with self.lock:
			self.val.value += 1

	def value(self):
		with self.lock:
			return self.val.value

# Plots min, max and mean + standard deviation bars of a population over time
def plot_line(xs, ys_population, title, filename):
	max_colour = 'rgb(0, 132, 180)'
	mean_colour = 'rgb(0, 172, 237)'
	std_colour = 'rgba(29, 202, 255, 0.2)'

	ys = torch.Tensor(ys_population)
	ys_min = ys.min(1)[0].squeeze()
	ys_max = ys.max(1)[0].squeeze()
	ys_mean = ys.mean(1).squeeze()
	ys_std = ys.std(1).squeeze()
	ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

	trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
	trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
	trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
	trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
	trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

	plotly.offline.plot({
		'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
		'layout': dict(title=title,
		 xaxis={'title': 'Step'},
		 yaxis={'title': 'Average'})
	}, filename=filename, auto_open=False)

def plot_loss(xs, p_loss, v_loss, title, filename):

	p_loss_color = 'rgb(0, 255, 0)'
	v_loss_color = 'rgb(255, 0, 0)'

	trace_p = Scatter(x=xs, y=p_loss, line=Line(color=p_loss_color), mode='lines', name='Policy Loss')
	trace_v = Scatter(x=xs, y=v_loss, line=Line(color=v_loss_color), mode='lines', name='Value Loss')

	plotly.offline.plot({
		'data': [trace_p, trace_v],
		'layout': dict(title=title,
		 xaxis={'title': 'Step'},
		 yaxis={'title': 'Average'})
	}, filename=filename, auto_open=False)

