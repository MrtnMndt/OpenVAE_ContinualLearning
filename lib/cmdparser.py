"""
Command line argument options parser.
Adopted and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py

Usage with two minuses "- -". Options are written with a minus "-" in command line, but
appear with an underscore "_" in the attributes' list.
"""

import argparse

parser = argparse.ArgumentParser(description='PyTorch Variational Training')

# Dataset and loading
parser.add_argument('--dataset', default='MNIST', help='name of dataset')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers. Default: 4')
parser.add_argument('-p', '--patch-size', default=28, type=int, help='patch size for crops. Default: 28')
parser.add_argument('--gray-scale', default=False, type=bool, help='use gray scale images. Default: False. '
                                                                   'If false, single channel images will be repeated '
                                                                   'to three channels.')
parser.add_argument('-noise', '--denoising-noise-value', default=0.25, type=float,
                    help='noise value for denoising. Float in range 0, 1. Default: 0.25')
parser.add_argument('-blur', '--blur', default=False, type=bool, help='turn on de-blurring. Default: False')

# Architecture and weight-init
parser.add_argument('-a', '--architecture', default='WRN', help='model architecture. Default: WRN')
parser.add_argument('--weight-init', default='kaiming-normal',
                    help='weight-initialization scheme. Default: kaiming-normal')
parser.add_argument('--wrn-depth', default=14, type=int,
                    help='amount of layers in the wide residual network. Default: 14')
parser.add_argument('--wrn-widen-factor', default=10, type=int,
                    help='width factor of the wide residual network. Default: 10')
parser.add_argument('--wrn-embedding-size', type=int, default=48,
                    help='number of output channels in the first wrn layer if widen factor is not being'
                         'applied to the first layer. Default: 48')
parser.add_argument('--double-wrn-blocks', type=bool, help='If turned on, uses 6 instead of 3 blocks'
                                                           ' and downsamples 6 times by factor 2. Should be used for'
                                                           'high resolution data, like flowers')

# data augmentation
parser.add_argument('-augment', '--data-augmentation', default=False, type=bool,
                    help='Turns on on the fly data augmentation. Default:False')
parser.add_argument('-hflip', '--hflip-p', default=0.0, type=float, help='probability of horizontal flip. Default 0.0')
parser.add_argument('-vflip', '--vflip-p', default=0.0, type=float, help='probability of vertical flip. Default 0.0')
parser.add_argument('-rand-trans', '--random-translation', default=False, type=bool,
                    help='turns on random translation by maximum 10 percent of the image size. Default: False')

# Training hyper-parameters
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run. Default: 120')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size. Default: 128')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate. Default: 1e-3')
parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='batch normalization. Default 1e-5')
parser.add_argument('-pf', '--print-freq', default=100, type=int, help='print frequency. Default: 100')

# Resuming training
parser.add_argument('--resume', default='', type=str, help='path to model to load/resume from. Default: none. '
                                                           'Also for stand-alone openset outlier evaluation script')

# Variational parameters
parser.add_argument('--var-latent-dim', default=60, type=int, help='Dimensionality of latent space. Default 60')
parser.add_argument('--var-beta', default=0.1, type=float, help='weight term for KLD loss. Default: 0.1')
parser.add_argument('--var-samples', default=1, type=int,
                    help='number of samples for the expectation in variational training. Default: 1')
parser.add_argument('--visualization-epoch', default=20, type=int, help='number of epochs after which generations/'
                                                                        'reconstructions are visualized/saved. '
                                                                        'Default: 20')

# Continual learning
parser.add_argument('--incremental-data', default=False, type=bool,
                    help='Convert dataloaders to class incremental ones. Default: False')
parser.add_argument('--train-incremental-upper-bound', default=False, type=bool,
                    help='Turn on class incremental training of upper bound baseline. Default: False')
parser.add_argument('--randomize-task-order', default=False, type=bool, help='randomizes the task order. '
                                                                             'Default: False')
parser.add_argument('--load-task-order', default='', type=str, help='path to numpy array specifying task order. '
                                                                    'Default: none')
parser.add_argument('--num-base-tasks', default=1, type=int,
                    help='Number of tasks to start with for incremental learning. Default: 1.'
                         'Maximum number of tasks has to be less than number of classes - 1.')
parser.add_argument('--num-increment-tasks', default=2, type=int, help='Number of task to add at once. Default: 2')
parser.add_argument('-genreplay', '--generative-replay', default=False, type=bool,
                    help='Turn on generative replay for data from old tasks. Default: False')

# Open set arguments
parser.add_argument('--openset-generative-replay', default=False, type=bool,
                    help='Turn on openset detection for generative replay. Default: False')
parser.add_argument('--openset-generative-replay-threshold', default=0.01, type=float,
                    help='Outlier probability threshold (float in range 0, 1. Default: 0.01')
parser.add_argument('--distance-function', default='cosine', help='Openset distance function. Default: cosine. '
                                                                  'choice of euclidean|cosine|mix')
parser.add_argument('-tailsize', '--openset-weibull-tailsize', default=0.05, type=float,
                    help='Tailsize in percent of data float in range 0, 1. Default: 0.05')

# Open set standalone script
parser.add_argument('--openset-datasets', default='FashionMNIST,AudioMNIST,KMNIST,CIFAR10,CIFAR100,SVHN',
                    help='name of openset datasets')
parser.add_argument('--percent-validation-outliers', default=0.05, type=float,
                    help='Assumed percentage of inherent outliers in the validation set of the original task. '
                         'Default 0.05, 5 percent. Is used to find priors and threshold values for testing.')
parser.add_argument('--calc-reconstruction', default=False, type=bool,
                    help='Turn on calculation of decoder. This option exists as calculating the decoder for multiple'
                         'samples can be computationally very expensive. Only turn this on if you are interested in'
                         'out-of-distribution detection according to reconstruction loss. Default: False')

# PixelVAE
parser.add_argument('--autoregression', default=False, type=bool, help='use PixelCNN decoder for generation')
parser.add_argument('--out-channels', default=60, type=int, help='number of output channels of decoder when'
                                                                 'autoregression is used. Default: 60')
parser.add_argument('--pixel-cnn-channels', default=60, type=int, help='num filters in PixelCNN convs. Default: 60')
parser.add_argument('--pixel-cnn-layers', default=3, type=int, help='number of PixelCNN layers. Default: 3')
parser.add_argument('--pixel-cnn-kernel-size', default=7, type=int, help='PixelCNN conv kernel size. Default: 7')

# IntroVAE
parser.add_argument('--introspection', default=False, type=bool, help='make use of introspection for training. '
                                                                      'Default: False')
parser.add_argument('--gamma', default=0.25, type=float, help='optional weight for gan-like KL losses. Default 0.25. '
                                                              'Empirically this factor does not seem to have '
                                                              'large impact')
parser.add_argument('--margin', default=100, type=float, help='margin to use for reconstructed and '
                                                              'fake kl losses. Default 100')
