from collections import OrderedDict
import math
import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=nn.functional.relu, downsample=False):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            nn.init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return nn.functional.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = nn.functional.avg_pool2d(h, 2)
        return h

class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=nn.functional.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(nn.functional.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return nn.functional.avg_pool2d(self.c2(h), 2)


class Discriminator(nn.Module):
    def __init__(self, num_colors, num_classes, nChannels, args):
        super(Discriminator, self).__init__()
        self.num_colors = num_colors
        self.batch_norm = args.batch_norm
        activation = nn.functional.relu
        self.activation = activation
        # self.nChannels = nChannels[:-1]
        # nFeatures = 16
        nFeatures = args.num_dis_feature
        self.nChannels = [nFeatures, nFeatures*2, nFeatures*4, nFeatures*8, nFeatures*16]
        self.block1 = OptimizedBlock(3, self.nChannels[0])
        self.block2 = Block(self.nChannels[0], self.nChannels[0],
                            activation=activation, downsample=True)
        self.block3 = Block(self.nChannels[0], self.nChannels[0],
                            activation=activation, downsample=False)
        self.block4 = Block(self.nChannels[0], args.var_latent_dim*16,
                            activation=activation, downsample=False)
        # self.block4 = Block(self.nChannels[2], self.nChannels[3],
        #                     activation=activation, downsample=True)
        # self.block5 = Block(self.nChannels[3], self.nChannels[4],
        #                     activation=activation, downsample=True)
        # self.block6 = Block(self.nChannels[4], args.var_latent_dim,
        #                     activation=activation, downsample=True)
        self.l7 = nn.utils.spectral_norm(nn.Linear(args.var_latent_dim*16, 1))
        if num_classes > 0:
            self.l_y = nn.utils.spectral_norm(
                nn.Embedding(num_classes, args.var_latent_dim*16))

        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            nn.init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 5):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        # print(h.shape)
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            if len(y.shape) == 1:
                emb =  self.l_y(y)
            else:
                emb = y
            output += torch.sum(emb * h, dim=1, keepdim=True)
        return output



	# 	self.discriminator_emb = nn.Sequential(OrderedDict([
	# 		('class Embedding',nn.Embedding(num_classes, args.var_latent_dim)),
	# 		]))
	# 	dim = 2

	# 	self.block = OrderedDict([
	# 		('discriminator_act1', nn.LeakyReLU(0.01)),
	# 		('discriminator_conv1', nn.utils.specral_norm(
	# 			nn.Conv2d(self.num_colors, self.nChannels[0], 4, 2, 1))
	# 		),
	# 		])
	# 	for i in range(1, len(self.nChannels)):
	# 		conv_name = 'discriminator_conv' + str(i+1)
	# 		act_name =  'discriminator_actv' + str(i+1)
	# 		self.block[act_name] = nn.LeakyReLU(0.01)
	# 		self.block[conv_name] = nn.utils.specral_norm(
	# 			nn.Conv2d(self.nChannels[i-1], self.nChannels[i], 4, 2, 1)
	# 			)
	# 		dim *= 2
	# 	self.block = nn.Sequential(self.block)
	# 	self.fc_block = nn.Sequential(OrderedDict([
	# 		('discriminator_latent', nn.Conv2d(self.nChannels[-1], args.var_latent_dim, args.patch_size//dim, 1, 0)),
	# 		]))
	# 	self.fc_cls = nn.Sequential(OrderedDict([
	# 		('discriminator_cls', nn.Linear(args.var_latent_dim, 1, bias=False)),
	# 		]))

	# def forward(self, x, y=None):
	# 	x = self.block(x)
	# 	x = self.fc_block(x)
	# 	x = x.squeeze()
	# 	if y is not None:
	# 		if len(y.shape) == 1:
	# 			y = self.discriminator_emb(y)
	# 		x += y
	# 	x = self.fc_cls(x)
	# 	return x

# from collections import OrderedDict
# import torch
# import torch.nn as nn
# class Discriminator(nn.Module):
# 	def __init__(self, num_colors, nChannels, args):
# 		super(Discriminator, self).__init__()
# 		self.num_colors = num_colors
# 		self.batch_norm = args.batch_norm
# 		self.nChannels = nChannels

# 		self.block = OrderedDict([
# 			('discriminator_conv1', nn.Conv2d(self.num_colors, self.nChannels[0], 4, 2, 1)),
# 			('discriminator_act1', nn.LeakyReLU(0.2)),
# 			('discriminator_conv2', nn.Conv2d(self.nChannels[0], self.nChannels[1] , 4, 2, 1)),
# 			('discriminator_bn1', nn.BatchNorm2d(self.nChannels[1],eps=self.batch_norm)),
# 			('discriminator_act2', nn.LeakyReLU(0.2)),
# 			])



# 		self.fc_input_dim = self.nChannels[1]*((args.patch_size//4)**2)
# 		self.fc_block = nn.Sequential(OrderedDict([
# 			('fc_discriminator_fc1',nn.Linear(self.fc_input_dim, 1024)),
# 			('fc_discriminator_bn1',nn.BatchNorm1d(1024)),
# 			('fc_discriminator_act1',nn.LeakyReLU(0.2)),
# 			('fc_discriminator_fc2',nn.Linear(1024, args.var_latent_dim, bias=False)),
# 			]))
# 		self.fc_cls = nn.Sequential(OrderedDict([
# 			('fc_discriminator_cls',nn.Linear(args.var_latent_dim, 1, bias=False)),
# 			]))


# 	def gradient_penalty(self, y, x):
# 		weight = torch.ones_like(y)
# 		gradient = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
# 		gradient = gradient.view(gradient.size(0),-1)		
# 		gradient = ((gradient.norm(2,1)-1) **2).mean()
# 		return self.lambda_gp * gradient

# 	def forward(self, x, y=None):
# 		x = self.block(x)
# 		x = x.view(-1,self.fc_input_dim)
# 		x = self.fc_block(x)
# 		if y is not None:
# 			x += y
# 		x = self.fc_cls(x)
# 		return x