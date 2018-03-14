

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torch.autograd import Variable



class SocialNet(nn.Module):
	def __init__(self, event=None, fashion=None, num_class=16):
		super(SocialNet, self).__init__()

		self.fashion = resnet34()
		self.event = resnet50()

		self.fc = nn.Sequential(
			nn.Linear(2048 + 512 + 512, 96),
			nn.Linear(96, num_class),
		)
		# self._initialize_weights(event, fashion)

	def forward(self, full, b1, b2):
		b1 = self.fashion(b1).data
		b2 = self.fashion(b2).data
		full = self.event(full).data

		features = torch.cat([b1, b2, full], 1)
		features = Variable(features.view(features.size(0), -1)).cuda()

		return self.fc(features)

	def _initialize_weights(self, event, fashion):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 1)
				# torch.nn.init.xavier_uniform(m.weight.data)
				m.bias.data.zero_()
			# elif isinstance(m, nn.Conv2d):
			# 	n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			# 	m.weight.data.normal_(0, np.sqrt(4. / n))
			# 	if m.bias is not None:
			# 		m.bias.data.zero_()
			# elif isinstance(m, nn.BatchNorm2d):
			# 	m.weight.data.fill_(1)
			# 	m.bias.data.zero_()

		self.event.load_weights(event)
		self.fashion.load_weights(fashion)




	# def save_checkpoint(self, state, ep, step, max_keep=15, path='/media/ramin/monster/models/sequence/'):
	# 	filename = os.path.join(path, 'ck-{0}-{1}.pth.tar'.format(ep, step))
	# 	torch.save(state, os.path.join(path, 'ck-last.path.tar'))
	# 	torch.save(state, filename)
	# 	def sorted_ls(path):
	# 			mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
	# 			return list(sorted(os.listdir(path), key=mtime))
	# 	files = sorted_ls(path)[:-max_keep]
	# 	for item in files:
	# 		os.remove(os.path.join(path, item))

	# def load_checkpoint(self, path='/media/ramin/monster/models/sequence/', filename=None):
	# 	if not filename:
	# 		filename = os.path.join(path, 'ck-last.path.tar')
	# 	else:
	# 		filename = os.path.join(path, filename)

	# 	self.load_state_dict(torch.load(filename))



class C_ResNet(ResNet):
	def __init__(self, block, layers, **kwargs):
		super(C_ResNet, self).__init__(block, layers, **kwargs)
		del self.fc

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		return x

	def load_weights(self, pretrained):
		self.load_state_dict(torch.load(pretrained)['state_dict'])



def resnet18(pretrained='', **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = C_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	if pretrained:  # weights.
		model.load_state_dict(torch.load(pretrained)['state_dict'])
	return model




def resnet34(pretrained='', **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = C_ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	if pretrained:  # weights.
		model.load_state_dict(torch.load(pretrained)['state_dict'])
	return model




def resnet50(pretrained='', **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = C_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:  # weights.
		model.load_state_dict(torch.load(pretrained)['state_dict'])
	return model







