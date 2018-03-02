from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform



class FashionDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, root_dir, mode='train', transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.dataset = self._load_dataset(mode)


	def __len__(self):
		return len(self.dataset)


	def _load_dataset(self, mode, cls_file='dataset/category.txt', bbox_file='dataset/bbox.txt', mode_file='dataset/partition.txt'):
		dataset = list()
		cls_file = open(cls_file, 'r')
		bbox_file = open(bbox_file, 'r')
		mode_file = open(mode_file, 'r')
		for line in cls_file:
			img, cls_idx = line.strip().split()
			bbox = [int(pos) for pos in bbox_file.readline().strip().split()[1:]]
			if mode_file.readline().strip().split()[1] == mode:
				dataset.append((img, int(cls_idx), bbox))
		return dataset


	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,
								self.dataset[idx][0])
		image = io.imread(img_name)
		# if self.transform:
		# 	box = self.dataset[idx][2]
		# 	if box[2] > image.shape[0]:
		# 		print(box[2], image.shape[0])
		# 		box[2] = image.shape[0]-1
		# 	if box[3] > image.shape[1]:
		# 		print(box[3], image.shape[1])
		# 		box[3] = image.shape[1]-1 
		# 	image = image[box[0]:box[2], box[1]:box[3]]
		# 	image = self.transform(image)
		if self.transform:
			box = self.dataset[idx][2]
			image = image[box[1]:box[3], box[0]:box[2]]
			image = self.transform(image)
		return image, self.dataset[idx][1]
