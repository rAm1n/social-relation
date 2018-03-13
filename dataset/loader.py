


from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import random
import os
import pickle
from itertools import izip


config = {

	'img_dir' : '/media/ramin/monster/dataset/social/PIPA-relation/imgs/',
	'body_dir' : '/media/ramin/monster/dataset/social/PIPA-relation/all_single_body/',
	'pair_dir' : 'socialRelation-vision/annotator_consistency3(used in our paper)/',
	'pair_pattern' : 'single_body{0}_{1}_{2}.txt', # pair_num (1,2) -  set (train , test) - 5,16
	'num_classes' : '16',
}



class Dataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, config, mode='train',  transform_body=None, transform_full=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.config
		self.transform_body = transform_body
		self.transform_full = transform_full

		self.dataset = self._load_dataset(mode)


	def __len__(self):
		return len(self.dataset)


	def _load_dataset(self, mode):

		DIR = self.config['pair_dir']

		pair_1 = self.config['pair_pattern'].format(1, mode, self.config['num_classes'])
		pair_2 = self.config['pair_pattern'].format(2, mode, self.config['num_classes'])

		pair_1 = os.path.join(DIR, pair_1)
		pair_2 = os.path.join(DIR, pair_2)


		pairs = list()

		with open(pair_1) as pair_1, open(pair_2) as pair_2:
			for x, y in izip(pair_1, pair_2):
				img_1, cls_1 = x.strip().split()
				img_2, cls_2 = y.strip().split()
				if cls_1 != cls_2:
					print('holy fuck!')
					print img_1, cls_1
					print img_2, cls_2
					exit()
				img_1 = os.path.join(self.config['body_dir'], img_1.split('/')[-1])
				img_2 = os.path.join(self.config['body_dir'], img_2.split('/')[-1])
				img = img_1.split('/')[-1][3:]

				pairs.append(img, img_1, img_2, int(cls_1))

		random.shuffle(pairs)
		return pairs


	def __getitem__(self, idx):
		img_full = io.imread(self.dataset[idx][0])
		img_1 = io.imread(self.dataset[idx][1])
		img_2 = io.imread(self.dataset[idx][2])
		cls = self.dataset[idx][3]

		if self.transform_full:
			img_full = self.transform_full(image)
		if self.transform_body:
			img_1 = self.transform_body(img_1)
			img_2 = self.transform_body(img_2)

		return [img_full, img_1, img_2, cls]
