

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck




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



def resnet18(pretrained='', **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = C_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	if pretrained:  # weights.
		model.load_state_dict(torch.load(pretrained))
	return model




def resnet34(pretrained='', **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = C_ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	if pretrained:  # weights.
		model.load_state_dict(torch.load(pretrained))
	return model




def resnet50(pretrained='', **kwargs):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = C_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:  # weights.
		model.load_state_dict(torch.load(pretrained))
	return model






