#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Jiale Chen'
__mail__ = 'chenjlcv@mail.ustc.edu.cn'

import cv2 as cv
import numpy as np
import random
import lmdb
import torch
from io import BytesIO
from PIL import Image
from chainer import iterators
from matplotlib import pyplot as plt
import scipy.misc

from chainer.dataset import dataset_mixin

class Dataset(dataset_mixin.DatasetMixin):
	'''
	Dataset 
	'''
	def __init__(self, lmdb_name, flip=None, gcn=None):
		self.env = lmdb.open(lmdb_name, readonly=True)
		self.txn = self.env.begin()
		self.gcn = gcn
		self.flip = flip
		print('Dataset has {} images.'.format(self.env.stat()['entries']))

	def horizontal_flip(self, image, _image):
		if random.random() < 0.5:
			image = image.transpose(Image.FLIP_LEFT_RIGHT)
			_image = _image.transpose(Image.FLIP_LEFT_RIGHT)
			return image, _image
		return image, _image

	def __len__(self):
		return self.env.stat()['entries']

	def get_example(self, idx):
		"""
		Extract images.
		"""
		key = ('%06d' % idx).encode('ascii')
		value = self.txn.get(key)
		bytes_io = BytesIO(value)
		image = Image.open(bytes_io)
		_image = image

		#image = self.apply_cropping(image)
		if self.flip:
			image, _image = self.horizontal_flip(image, _image)

		if self.gcn:
			image = np.array(image)/127.5 - 1.
			_image = image

		image = cv.resize(np.array(image), None, fx=1.0/8.0, fy=1.0/8.0, interpolation=cv.INTER_NEAREST)

		image = np.transpose(image, (2,0,1))
		_image = np.transpose(_image, (2,0,1))

		return image, _image

	def __del__(self):
		self.env.close()

def batch2feeds(batch):
	"""
	Args:
	  batch: a batch that is returned by dataset_iterator,
	  which is a list of tuples (image)
	Return:
		images: float32 array [batch_size x H x W x C]
	"""
	images, _images = zip(*batch)
	images = torch.FloatTensor(np.asarray(images))
	_images = torch.FloatTensor(np.asarray(_images))

	return images, _images

def save_obs(obs, res_obs, filename):
	height, width, channel = obs.shape
	
	combined_imgs = np.ones((2*height, width, channel))
	for i in range(height):
		for j in range(width):
			combined_imgs[i, j, :] = obs[i,j,:]
			combined_imgs[height+i, j, :] = res_obs[i,j,:]

	print('Saving the img observation.', filename)
	scipy.misc.imsave(filename, combined_imgs)

def save_grad(grad_first, grad_last, filename):
	height, width, channel = grad_last.shape # (160, 128, 3)

	grad_first = cv.resize(grad_first, (128,160))
	print(grad_first.shape)

	combined_imgs = np.ones((2*height, width, channel))
	for i in range(height):
		for j in range(width):
			combined_imgs[i, j, :] = grad_last[i,j,:]
			combined_imgs[height+i, j, :] = grad_first[i,j,:]

	print('Saving the gradients.', filename)
	scipy.misc.imsave(filename, combined_imgs)

if __name__ == '__main__':
	dataset = Dataset('/home/chenjl/Research/Face_restoration/datasets/celeba_eval', flip=False, gcn=True, downsample=True)
	img, _ = dataset.get_example(2)
