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
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--patch-size', default=28, type=int, help='patch size for crops (default: 28)')
parser.add_argument('--gray-scale', default=False, type=bool, help='use gray scale images (default: False). '
                                                                   'If false, single channel images will be repeated '
                                                                   'to three channels.')
parser.add_argument('-noise', '--denoising-noise-value', default=0.25, type=float,
                    help='noise value for denoising. (float in range [0, 1]. Default: 0.25)')

# Architecture and weight-init
parser.add_argument('-a', '--architecture', default='WRN', help='model architecture (default: WRN)')
parser.add_argument('--weight-init', default='kaiming-normal',
                    help='weight-initialization scheme (default: kaiming-normal)')
parser.add_argument('--wrn-depth', default=16, type=int,
                    help='amount of layers in the wide residual network (default: 16)')
parser.add_argument('--wrn-widen-factor', default=10, type=int,
                    help='width factor of the wide residual network (default: 10)')
parser.add_argument('--wrn-embedding-size', type=int, default=48,
                    help='number of output channels in the first wrn layer if widen factor is not being'
                         'applied to the first layer (default: 48)')

# Training hyper-parameters
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate (default: 1e-3)')
parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='batch normalization (default 1e-5)')
parser.add_argument('-pf', '--print-freq', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('-log', '--log-weights', default=False, type=bool,
                    help='Log weights and gradients to TensorBoard (default: False)')

# Resuming training
parser.add_argument('--resume', default='', type=str, help='path to model to load/resume from(default: none). '
                                                           'Also for stand-alone openset outlier evaluation script')

# Variational parameters
parser.add_argument('--var-latent-dim', default=60, type=int, help='Dimensionality of latent space')
parser.add_argument('--var-beta', default=0.1, type=float, help='weight term for KLD loss (default: 0.1)')
parser.add_argument('--var-samples', default=1, type=int,
                    help='number of samples for the expectation in variational training (default: 1)')
parser.add_argument('--visualization-epoch', default=20, type=int, help='number of epochs after which generations/'
                                                                        'reconstructions are visualized/saved'
                                                                        '(default: 20)')

# Continual learning
parser.add_argument('--incremental-data', default=False, type=bool,
                    help='Convert dataloaders to class incremental ones')
parser.add_argument('--train-incremental-upper-bound', default=False, type=bool,
                    help='Turn on class incremental training of upper bound baseline')
parser.add_argument('--randomize-task-order', default=False, type=bool, help='randomizes the task order')
parser.add_argument('--load-task-order', default='', type=str, help='path to numpy array specifying task order')
parser.add_argument('--num-base-tasks', default=1, type=int,
                    help='Number of tasks to start with for incremental learning (default: 1).'
                         'Maximum number of tasks has to be less than number of classes - 1.')
parser.add_argument('--num-increment-tasks', default=2, type=int, help='Number of task to add at once')

parser.add_argument('--cross-dataset', type=bool, default=False, help='do cross-dataset CL (default: False)')
parser.add_argument('--dataset-order', default='AudioMNIST, MNIST, FashionMNIST', type=str,
                    help='dataset order in cross-dataset CL (default: "AudioMNIST, MNIST, FashionMNIST")')

parser.add_argument('-genreplay', '--generative-replay', default=False, type=bool,
                    help='Turn on generative replay for data from old tasks')

# Open set arguments
parser.add_argument('--openset-generative-replay', default=False, type=bool,
                    help='Turn on openset detection for generative replay')
parser.add_argument('--openset-generative-replay-threshold', default=0.01, type=float,
                    help='Outlier probability threshold (float in range [0, 1]. Default: 0.01)')
parser.add_argument('--distance-function', default='cosine', help='Openset distance function (default: cosine) '
                                                                  'choice of euclidean|cosine|mix')
parser.add_argument('-tailsize', '--openset-weibull-tailsize', default=0.05, type=float,
                    help='Tailsize in percent of data (float in range [0, 1]. Default: 0.05')

# Open set standalone script
parser.add_argument('--openset-datasets', default='FashionMNIST,AudioMNIST,KMNIST,CIFAR10,CIFAR100,SVHN',
                    help='name of openset datasets')
parser.add_argument('--percent-validation-outliers', default=0.05, type=float,
                    help='Assumed percentage of inherent outliers in the validation set of the original task. '
                         'default (0.05 -> 5%). Is used to find priors and threshold values for testing.')
parser.add_argument('--calc-reconstruction', default=False, type=bool,
                    help='Turn on calculation of decoder. This option exists as calculating the decoder for multiple'
                         'samples can be computationally very expensive. Only turn this on if you are interested in'
                         'out-of-distribution detection according to reconstruction loss.')

# PixelVAE
parser.add_argument('--autoregression', default=False, type=bool, help='use PixelCNN decoder for generation')
parser.add_argument('--out-channels', default=60, type=int, help='number of output channels of decoder when'
                                                                 'autoregression is used (default: 60)')
parser.add_argument('--pixel-cnn-channels', default=60, type=int, help='num filters in PixelCNN convs (default: 60)')
parser.add_argument('--pixel-cnn-layers', default=3, type=int, help='number of PixelCNN layers (default: 3)')
parser.add_argument('--pixel-cnn-kernel-size', default=7, type=int, help='PixelCNN conv kernel size (default: 7)')

# Debug
parser.add_argument('--debug','-d',action = 'store_true', help = 'pdb enable')

# No Recon
parser.add_argument("--no-recon", action = 'store_true', help='no recon loss')

# WGAN-GP
parser.add_argument('--gan', action = 'store_true', help ='train generator as gan fashion')
parser.add_argument('--feature-wise-loss', action = "store_true")
parser.add_argument('--gan-epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('-lrG', '--learning-rate-G', default=2e-4, type=float)
parser.add_argument('-lrD', '--learning-rate-D', default=2e-4, type=float)
parser.add_argument('-rg','--ratio-G', default=1, type=int)
parser.add_argument('-rd','--ratio-D', default=2, type=int)
parser.add_argument('--lambda-gp', default = 10, type =float, help ='penalty')
parser.add_argument('--encoder-dist', action='store_true', help ='feature_distance loss')
