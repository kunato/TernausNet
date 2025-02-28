
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import torchvision
import torch.nn.functional as F

class PyramidPool(nn.Module):

	def __init__(self, in_features, out_features, pool_size):
		super(PyramidPool,self).__init__()

		self.features = nn.Sequential(
			nn.AdaptiveAvgPool2d(pool_size),
			nn.Conv2d(in_features, out_features, 1, bias=False),
			nn.BatchNorm2d(out_features, momentum=.95),
			nn.ReLU(inplace=True)
		)


	def forward(self, x):
		size=x.size()
		output=F.upsample(self.features(x), size[2:], mode='bilinear')
		return output

class PSPNet34(nn.Module):

    def __init__(self, num_classes, pretrained = True):
        super(PSPNet34,self).__init__()
        print("initializing model")

        self.resnet = torchvision.models.resnet34(pretrained = pretrained)

        self.layer5a = PyramidPool(512, 128, 1)
        self.layer5b = PyramidPool(512, 128, 2)
        self.layer5c = PyramidPool(512, 128, 3)
        self.layer5d = PyramidPool(512, 128, 6)
		
        self.final = nn.Sequential(
        	nn.Conv2d(1024, 256, 3, padding=1, bias=False),
        	nn.BatchNorm2d(256, momentum=.95),
        	nn.ReLU(inplace=True),
        	nn.Dropout(.1),
        	nn.Conv2d(256, num_classes, 1),
        )

        initialize_weights(self.layer5a,self.layer5b,self.layer5c,self.layer5d,self.final)




    def forward(self, x):
        count=0

        size=x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.final(torch.cat([
        	x,
        	self.layer5a(x),
        	self.layer5b(x),
        	self.layer5c(x),
        	self.layer5d(x),
        ], 1))


        return F.sigmoid(F.upsample_bilinear(x,size[2:]))




class PSPNet50(nn.Module):

    def __init__(self, num_classes, pretrained = True):
        super(PSPNet50,self).__init__()
        print("initializing model")

        self.resnet = torchvision.models.resnet50(pretrained = pretrained)


        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)

        self.final = nn.Sequential(
        	nn.Conv2d(4096, 512, 3, padding=1, bias=False),
        	nn.BatchNorm2d(512, momentum=.95),
        	nn.ReLU(inplace=True),
        	nn.Dropout(.1),
        	nn.Conv2d(512, num_classes, 1),
        )

        initialize_weights(self.layer5a,self.layer5b,self.layer5c,self.layer5d,self.final)




    def forward(self, x):
        count=0

        size=x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.final(torch.cat([
        	x,
        	self.layer5a(x),
        	self.layer5b(x),
        	self.layer5c(x),
        	self.layer5d(x),
        ], 1))


        return F.sigmoid(F.upsample_bilinear(x,size[2:]))

def initialize_weights(*models):
	for model in models:
		for module in model.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_normal(module.weight)
				if module.bias is not None:
					module.bias.data.zero_()
			elif isinstance(module, nn.BatchNorm2d):
				module.weight.data.fill_(1)
				module.bias.data.zero_()