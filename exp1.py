import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from dataset.fashion import FashionDataset
from dataset.loader import Dataset
from models import *
from utils import config


parser = argparse.ArgumentParser(description='Social recognition')

parser.add_argument('--fashion', default='weights/resnet34-fashion-12cls-adam-3e-4.pth.tar', metavar='DIR',
					help='path to dataset')
parser.add_argument('--event', default='weights/resnet50_event_acc74.pth.tar', metavar='DIR',
					help='path to dataset')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')



best_prec1 = 0


def main():
	# global args, best_prec1, model, train_dataset, val_loader
	args = parser.parse_args()

	args.distributed = args.world_size > 1

	if args.distributed:
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size)


	model = SocialNet(fashion=args.fashion, event=args.event, num_class=config['num_class']).cuda()

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()

#	optimizer = torch.optim.SGD(model.parameters(), args.lr,
#								momentum=args.momentum,
#								weight_decay=args.weight_decay)
	optimizer = torch.optim.Adam(model.fc.parameters(), args.lr,
								betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	cudnn.benchmark = True

	# Data loading code

	train_dataset = Dataset(config, mode='train')
	test_dataset = Dataset(config, mode='test')


	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))


	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)


	if args.evaluate:
		validate(val_loader, model, criterion)
		return

	for epoch in range(args.start_epoch, args.epochs):

		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		# evaluate on validation set
		prec1 = validate(test_loader, model, criterion)

		# remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer' : optimizer.state_dict(),
		}, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (img, img_1, img_2, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		img_var = torch.autograd.Variable(img)
		img_1_var = torch.autograd.Variable(img_1)
		img_2_var = torch.autograd.Variable(img_2)
		target_var = torch.autograd.Variable(target)

		# compute output
		output = model(img_var, img_1_var, img_2_var, target_var)

		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec5, correct, count  = accuracy(output.data, target, \
			topk=(1, 5), num_cls=train_loader.dataset.config['num_class'])


		losses.update(loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   epoch, i, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()
	end = time.time()

	correct = list()
	count = list()

	for i, (img, img_1, img_2, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		img_var = torch.autograd.Variable(img, volatile=True)
		img_1_var = torch.autograd.Variable(img_1, volatile=True)
		img_2_var = torch.autograd.Variable(img_2, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(img_var, img_1_var, img_2_var, target_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec5, correct, count  = accuracy(output.data, target, \
			topk=(1, 5), num_cls=train_loader.dataset.config['num_class'])


		correct.append(correct_batch)
		count.append(count_batch)

		losses.update(loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   i, len(val_loader), batch_time=batch_time, loss=losses,
				   top1=top1, top5=top5))

	print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
		  .format(top1=top1, top5=top5))

	correct = np.array(correct).sum(0)
	count = np.array(count).sum(0).astype(np.int32)
	pres = correct/count

	for idx , item in enumerate(pres):
		print idx, count[idx], item

	return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,), num_cls=46):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k]
		res.append(correct_k.view(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size))

	cls_correct = np.zeros(num_cls)
	cls_count = np.zeros(num_cls)
	for idx, tar in enumerate(target):
		if correct[0,idx]:
			cls_correct[target[idx]] += 1
		cls_count[target[idx]] += 1

	res.append(cls_correct)
	res.append(cls_count)

	return res


if __name__ == '__main__':
	main()
