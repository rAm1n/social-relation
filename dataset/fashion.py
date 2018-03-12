from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import random
import os



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
		self.class_to_idx = dict()
		self.dataset = self._load_dataset(mode)


	def __len__(self):
		return len(self.dataset)


# 	def _load_dataset(self, mode, cls_file='dataset/category.txt', bbox_file='dataset/bbox.txt', mode_file='dataset/partition.txt', min_sample=500):
# 		dataset = list()
# 		cls_len = dict()
# 		cls_file = open(cls_file, 'r')
# 		bbox_file = open(bbox_file, 'r')
# 		mode_file = open(mode_file, 'r')
# 		if 'train' not in mode:
# 			with open('dataset/clasess.txt','r') as f:
# 				for line in f:
# 					line = line.strip().split(',')
# 					self.class_to_idx[line[0]] = int(line[1])

# 		for line in cls_file:
# 			img, cls = line.strip().split()
# 			bbox = [int(pos) for pos in bbox_file.readline().strip().split()[1:]]
# 			if mode_file.readline().strip().split()[1] in mode:
# #				if cls_idx not in self.class_to_idx:
# #					self.class_to_idx[cls_idx] = len(self.class_to_idx)
# #					self.idx_to_class[len(self.class_to_idx)] = cls_idx
# #					self.idx_len[len(selef.class_to_idx)] = 0
# #					self.classes.append(cls_idx)
# 				if 'train' in mode:
# 					dataset.append((img, cls, bbox))
# 					if cls not in cls_len:
# 						cls_len[cls] =0
# 					cls_len[cls] += 1
# 				else:
# 					if cls in self.class_to_idx:
# 						dataset.append((img, self.class_to_idx[cls], bbox))

# 		if 'train' in mode:
# 			for cls in cls_len:
# 				if cls_len[cls] > min_sample:
# 					self.class_to_idx[cls] = len(self.class_to_idx)

# 			dataset = [[item[0], self.class_to_idx[item[1]], item[2]] for item in dataset if (item[1] in self.class_to_idx.keys())]

# 			with open('dataset/clasess.txt','w') as f:
# 				for cls in self.class_to_idx:
# 					f.write('{0},{1}\n'.format(cls, self.class_to_idx[cls]))


# 		random.shuffle(dataset)
# 		return dataset


	def _load_dataset(self, mode, cls_file='dataset/category.txt', bbox_file='dataset/bbox.txt', min_sample=500):
		dataset_file = 'dataset-{0}.pkl'.format(min_sample)
		if os.path.isfile(dataset_file):
			with open(dataset_file, 'r') as f:
				dataset = pickle.load(f)
			return dataset[mode]

		tmp = list()
		dataset dict()
		cls_len = dict()

		cls_file = open(cls_file, 'r')
		bbox_file = open(bbox_file, 'r')
		mode_file = open(mode_file, 'r')


		for line in cls_file:
			img, cls = line.strip().split()
			bbox = [int(pos) for pos in bbox_file.readline().strip().split()[1:]]
			tmp.append((img, cls, bbox))
			if cls not in cls_len:
				cls_len[cls] = 0
			cls_len[cls] += 1

		random.shuffle(tmp)

		for cls in cls_len:
			if cls_len[cls] > min_sample:
				self.class_to_idx[cls] = len(self.class_to_idx)

		dataset['train'] = tmp[:len(tmp) * 0.85]
		dataset['test'] = tmp[len(tmp) * 0.85: ]

		for key in dataset:
			d = list()
			for item in dataset[key]:
				if (item[1] in self.class_to_idx.keys())
					d.append([item[0], self.class_to_idx[item[1]], item[2]])
			dataset[key] = d

		with open(dataset_file, 'w') as f:
			pickle.dump(dataset, f)

		return dataset[mode]



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
