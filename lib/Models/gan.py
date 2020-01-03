from collections import OrderedDict
import torch
import torch.nn as nn
class Gan_Model(nn.Module):
	def __init__(self, device, num_classes, num_colors, args):
		super(Gan_Model, self).__init__()
		self.device = device
		self.num_colors = num_colors
		self.lambda_gp = args.lambda_gp
		self.widen_factor = args.wrn_widen_factor
		self.batch_norm = args.batch_norm
		self.nChannels = [args.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor]

		self.block = nn.Sequential(OrderedDict([
			('discriminator_conv1', nn.Conv2d(self.num_colors, self.nChannels[0], 4, 2, 1)),
			('discriminator_act1', nn.LeakyReLU(0.2)),
			('discriminator_conv2', nn.Conv2d(self.nChannels[0], self.nChannels[1] , 4, 2, 1)),
			('discriminator_bn1', nn.BatchNorm2d(self.nChannels[1],eps=self.batch_norm)),
			('discriminator_act2', nn.LeakyReLU(0.2)),
			]))
		self.fc_input_dim = self.nChannels[1]*((args.patch_size//4)**2)
		self.fc_block = nn.Sequential(OrderedDict([
			('fc_discriminator_fc1',nn.Linear(self.fc_input_dim, 1024)),
			('fc_discriminator_bn1',nn.BatchNorm1d(1024)),
			('fc_discriminator_act1',nn.LeakyReLU(0.2)),
			('fc_discriminator_fc2',nn.Linear(1024,1)),
			]))


	def gradient_penalty(self, y, x):
		weight = torch.ones_like(y)
		gradient = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
		gradient = gradient.view(gradient.size(0),-1)		
		gradient = ((gradient.norm(2,1)-1) **2).mean()
		return self.lambda_gp * gradient

	def forward(self, x):
		x = self.block(x)
		x = x.view(-1,self.fc_input_dim)
		x = self.fc_block(x)
		return x
