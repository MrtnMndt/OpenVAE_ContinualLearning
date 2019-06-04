from collections import OrderedDict
import torch
import torch.nn as nn

# Note: nn.Upsample is going to be deprecated in favor of nn.functional.interpolate in future torch versions!


def grow_classifier(classifier, class_increment, weight_initializer):
    """
    Function to grow the units of a classifier an initializing only the newly added units while retaining old knowledge.

    Parameters:
        classifier (torch.nn.module): Trained classifier portion of the model.
        class_increment (int): Number of classes/units to add.
        weight_initializer (WeightInit): Weight initializer class instance defining initialization schemes/functions.
    """

    # add the corresponding amount of features and resize the weights
    classifier[-1].out_features += class_increment
    # we can safely resize the tensor because the change is in the first dimension
    # in this case an empty row is simply added (think of flattened arrays). This wouldn't be the case for dimension 2
    classifier[-1].weight.data.resize_(classifier[-1].weight.data.size(0) + class_increment,
                                       classifier[-1].weight.data.size(1))

    # if gradients aren't none (i.e. because they have been deleted), resize the gradients correspondingly.
    # As gradients are set to zero before each optimization step we needn't worry about initialization for them.
    if not isinstance(classifier[-1].weight.grad, type(None)):
        # no need to initialize as gradients are set to zero before each training step
        classifier[-1].weight.grad.data.resize_(classifier[-1].weight.grad.data.size(0) + class_increment,
                                                classifier[-1].weight.grad.data.size(1))

    # check if there is a bias and resize those values too
    if not isinstance(classifier[-1].bias, type(None)):
        classifier[-1].bias.data.resize_(classifier[-1].bias.data.size(0) + class_increment)[:-class_increment] = 0
        if not isinstance(classifier[-1].bias.grad, type(None)):
            # no need to initialize as gradients are set to zero before each training step
            classifier[-1].bias.grad.data.resize_(classifier[-1].bias.grad.data.size(0) + class_increment)

    # for the second dimension of the parameters we can't simply resize because the flattened array is going to shift
    # values in a wrong fashion. We thus first create a temporary copy, resize the tensor and copy the values
    # back after initialization.
    tmp_weights = classifier[-1].weight.data[0:-class_increment, :].clone()
    if not isinstance(classifier[-1].bias, type(None)):
        tmp_bias = classifier[-1].bias.data[0:-class_increment].clone()

    # initialize the correctly shaped layer.
    weight_initializer.layer_init(classifier[-1])

    # copy back the temporarily saved parameters for the slice of previously trained classes.
    classifier[-1].weight.data[0:-class_increment, :] = tmp_weights
    if not isinstance(classifier[-1].bias, type(None)):
        classifier[-1].bias.data[0:-class_increment] = tmp_bias


def get_feat_size(block, spatial_size, ncolors=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.

    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    """

    x = torch.randn(2, ncolors, spatial_size, spatial_size)
    out = block(x)
    num_feat = out.size(1)
    spatial_dim_x = out.size(2)
    spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y


class SingleConvLayer(nn.Module):
    """
    Convenience function defining a single block consisting of a convolution or transposed convolution followed by
    batch normalization and a rectified linear unit activation function.
    """
    def __init__(self, l, fan_in, fan_out, kernel_size=3, padding=1, stride=1, batch_norm=1e-5, is_transposed=False):
        super(SingleConvLayer, self).__init__()

        if is_transposed:
            self.layer = nn.Sequential(OrderedDict([
                ('transposed_conv' + str(l), nn.ConvTranspose2d(fan_in, fan_out, kernel_size=kernel_size,
                                                                padding=padding, stride=stride, bias=False))
            ]))
        else:
            self.layer = nn.Sequential(OrderedDict([
                ('conv' + str(l), nn.Conv2d(fan_in, fan_out, kernel_size=kernel_size, padding=padding, stride=stride,
                                            bias=False))
            ]))
        if batch_norm > 0.0:
            self.layer.add_module('bn' + str(l), nn.BatchNorm2d(fan_out, eps=batch_norm))

        self.layer.add_module('act' + str(l), nn.ReLU())

    def forward(self, x):
        x = self.layer(x)
        return x


class SingleLinearLayer(nn.Module):
    """
    Convenience function defining a single block consisting of a fully connected (linear) layer followed by
    batch normalization and a rectified linear unit activation function.
    """
    def __init__(self, l, fan_in, fan_out, batch_norm=1e-5):
        super(SingleLinearLayer, self).__init__()

        self.fclayer = nn.Sequential(OrderedDict([
            ('fc' + str(l), nn.Linear(fan_in, fan_out, bias=False)),
        ]))

        if batch_norm > 0.0:
            self.fclayer.add_module('bn' + str(l), nn.BatchNorm1d(fan_out, eps=batch_norm))

        self.fclayer.add_module('act' + str(l), nn.ReLU())

    def forward(self, x):
        x = self.fclayer(x)
        return x


class MLP(nn.Module):
    """
    MLP design with two hidden layers and 400 hidden units each in the encoder according to
    ﻿Measuring Catastrophic Forgetting in Neural Networks: https://arxiv.org/abs/1708.02072
    Extended to the variational setting and our unified model.
    """

    def __init__(self, device, num_classes, num_colors, args):
        super(MLP, self).__init__()

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.device = device
        self.out_channels = args.out_channels

        self.seen_tasks = []

        self.num_samples = args.var_samples
        self.latent_dim = args.var_latent_dim

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_layer1', SingleLinearLayer(1, self.num_colors * (self.patch_size ** 2), 400,
                                                 batch_norm=self.batch_norm)),
            ('encoder_layer2', SingleLinearLayer(2, 400, 400, batch_norm=self.batch_norm))
        ]))

        self.latent_mu = nn.Linear(400, self.latent_dim, bias=False)
        self.latent_std = nn.Linear(400, self.latent_dim, bias=False)

        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_layer0', SingleLinearLayer(0, self.latent_dim, 400, batch_norm=self.batch_norm)),
            ('decoder_layer1', SingleLinearLayer(1, 400, 400, batch_norm=self.batch_norm)),
            ('decoder_layer2', nn.Linear(400, self.out_channels * (self.patch_size ** 2), bias=False))
        ]))

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        z_mean = self.latent_mu(x)
        z_std = self.latent_std(x)
        return z_mean, z_std

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        x = self.decoder(z)
        x = x.view(-1, self.out_channels, self.patch_size, self.patch_size)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        x = x.view(-1, self.out_channels, self.patch_size, self.patch_size)
        return x

    def forward(self, x):
        z_mean, z_std = self.encode(x)
        output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
                                     self.patch_size).to(self.device)
        classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
        for i in range(self.num_samples):
            z = self.reparameterize(z_mean, z_std)
            output_samples[i] = self.decode(z)
            classification_samples[i] = self.classifier(z)
        return classification_samples, output_samples, z_mean, z_std


class DCNN(nn.Module):
    """
    CNN architecture inspired by WAE-DCGAN from https://arxiv.org/pdf/1511.06434.pdf but without the GAN component.
    Extended to the variational setting and to our unified model.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(DCNN, self).__init__()

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.device = device
        self.out_channels = args.out_channels

        # for 28x28 images, e.g. MNIST. We set the innermost convolution's kernel from 4 to 3 and adjust the
        # paddings in the decoder to upsample correspondingly. This way the incoming spatial dimensionality
        # to the latent space stays the same as with 32x32 resolution
        self.inner_kernel_size = 4
        self.inner_padding = 0
        self.outer_padding = 1
        if args.patch_size < 32:
            self.inner_kernel_size = 3
            self.inner_padding = 1
            self.outer_padding = 0

        self.seen_tasks = []

        self.num_samples = args.var_samples
        self.latent_dim = args.var_latent_dim

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_layer1', SingleConvLayer(1, self.num_colors, 128, kernel_size=4, stride=2, padding=1,
                                               batch_norm=self.batch_norm)),
            ('encoder_layer2', SingleConvLayer(2, 128, 256, kernel_size=4, stride=2, padding=1,
                                               batch_norm=self.batch_norm)),
            ('encoder_layer3', SingleConvLayer(3, 256, 512, kernel_size=4, stride=2, padding=1,
                                               batch_norm=self.batch_norm)),
            ('encoder_layer4', SingleConvLayer(4, 512, 1024, kernel_size=self.inner_kernel_size, stride=2, padding=0,
                                               batch_norm=self.batch_norm))
        ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,
                                                                                          self.num_colors)
        self.latent_mu = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                                   self.latent_dim, bias=False)
        self.latent_std = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                                    self.latent_dim, bias=False)

        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

        self.latent_decoder = SingleLinearLayer(0, self.latent_dim, self.enc_spatial_dim_x * self.enc_spatial_dim_y *
                                                self.enc_channels, batch_norm=self.batch_norm)

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_layer1', SingleConvLayer(1, 1024, 512, kernel_size=4, stride=2, padding=self.inner_padding,
                                               batch_norm=self.batch_norm, is_transposed=True)),
            ('decoder_layer2', SingleConvLayer(2, 512, 256, kernel_size=4, stride=2, padding=self.outer_padding,
                                               batch_norm=self.batch_norm, is_transposed=True)),
            ('decoder_layer3', SingleConvLayer(3, 256, 128, kernel_size=4, stride=2, padding=self.outer_padding,
                                               batch_norm=self.batch_norm, is_transposed=True)),
            ('decoder_layer4', nn.ConvTranspose2d(128, self.out_channels, kernel_size=4, stride=2,
                                                  padding=1, bias=False))
        ]))

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z_mean = self.latent_mu(x)
        z_std = self.latent_std(x)
        return z_mean, z_std

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        z = self.latent_decoder(z)
        z = z.view(z.size(0), self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        z_mean, z_std = self.encode(x)
        output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
                                     self.patch_size).to(self.device)
        classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
        for i in range(self.num_samples):
            z = self.reparameterize(z_mean, z_std)
            output_samples[i] = self.decode(z)
            classification_samples[i] = self.classifier(z)
        return classification_samples, output_samples, z_mean, z_std


class WRNBasicBlock(nn.Module):
    """
    Convolutional or transposed convolutional block consisting of multiple 3x3 convolutions with short-cuts,
    ReLU activation functions and batch normalization.
    """
    def __init__(self, in_planes, out_planes, stride, batchnorm=1e-5, is_transposed=False):
        super(WRNBasicBlock, self).__init__()

        if is_transposed:
            self.layer1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                             output_padding=int(stride > 1), bias=False)
        else:
            self.layer1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.useShortcut = ((in_planes == out_planes) and (stride == 1))
        if not self.useShortcut:
            if is_transposed:
                self.shortcut = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                                                   output_padding=int(1 and stride == 2), bias=False)
            else:
                self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        if not self.useShortcut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.layer1(out if self.useShortcut else x)))
        out = self.conv2(out)

        return torch.add(x if self.useShortcut else self.shortcut(x), out)


class WRNNetworkBlock(nn.Module):
    """
    Convolutional or transposed convolutional block
    """
    def __init__(self, nb_layers, in_planes, out_planes, block_type, batchnorm=1e-5, stride=1,
                 is_transposed=False):
        super(WRNNetworkBlock, self).__init__()

        if is_transposed:
            self.block = nn.Sequential(OrderedDict([
                ('convT_block' + str(layer + 1), block_type(layer == 0 and in_planes or out_planes, out_planes,
                                                             layer == 0 and stride or 1, batchnorm=batchnorm,
                                                             is_transposed=(layer == 0)))
                for layer in range(nb_layers)
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('conv_block' + str(layer + 1), block_type((layer == 0 and in_planes) or out_planes, out_planes,
                                                           (layer == 0 and stride) or 1, batchnorm=batchnorm))
                for layer in range(nb_layers)
            ]))

    def forward(self, x):
        x = self.block(x)
        return x


class WRN(nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting and to our unified model.
    """
    def __init__(self, device, num_classes, num_colors, args):
        super(WRN, self).__init__()

        self.widen_factor = args.wrn_widen_factor
        self.depth = args.wrn_depth

        self.batch_norm = args.batch_norm
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.device = device
        self.out_channels = args.out_channels

        self.seen_tasks = []

        self.num_samples = args.var_samples
        self.latent_dim = args.var_latent_dim

        self.nChannels = [args.wrn_embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,
                          64 * self.widen_factor]

        assert ((self.depth - 4) % 6 == 0)
        self.num_block_layers = int((self.depth - 4) / 6)

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
            ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
                                               WRNBasicBlock, batchnorm=self.batch_norm)),
            ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
                                               WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
            ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
                                               WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
            ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
            ('encoder_act1', nn.ReLU(inplace=True))
        ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,
                                                                                          self.num_colors)
        self.latent_mu = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels,
                                   self.latent_dim, bias=False)
        self.latent_std = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                                    self.latent_dim, bias=False)

        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

        self.latent_decoder = nn.Linear(self.latent_dim, self.enc_spatial_dim_x * self.enc_spatial_dim_y *
                                        self.enc_channels, bias=False)

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[2],
                                               WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
            ('decoder_upsample1', nn.Upsample(scale_factor=2, mode='nearest')),
            ('decoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[1],
                                               WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
            ('decoder_upsample2', nn.Upsample(scale_factor=2, mode='nearest')),
            ('decoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[0],
                                               WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
            ('decoder_bn1', nn.BatchNorm2d(self.nChannels[0], eps=self.batch_norm)),
            ('decoder_act1', nn.ReLU(inplace=True)),
            ('decoder_conv1', nn.Conv2d(self.nChannels[0], self.out_channels, kernel_size=3, stride=1, padding=1,
                                        bias=False))
        ]))

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z_mean = self.latent_mu(x)
        z_std = self.latent_std(x)
        return z_mean, z_std

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        z = self.latent_decoder(z)
        z = z.view(z.size(0), self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        x = self.decoder(z)
        return x

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        z_mean, z_std = self.encode(x)
        output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
                                     self.patch_size).to(self.device)
        classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
        for i in range(self.num_samples):
            z = self.reparameterize(z_mean, z_std)
            output_samples[i] = self.decode(z)
            classification_samples[i] = self.classifier(z)
        return classification_samples, output_samples, z_mean, z_std
