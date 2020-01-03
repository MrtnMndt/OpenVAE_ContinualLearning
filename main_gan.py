########################
# Importing libraries
########################
# System libraries
import os
import random
from time import gmtime, strftime
import numpy as np
import pickle
import copy
import pdb

# Tensorboard for PyTorch logging and visualization
from torch.utils.tensorboard import SummaryWriter

# Torch libraries
import torch
import torch.backends.cudnn as cudnn

# Custom library
import lib.Models.architectures as architectures
from lib.Models.gan import Gan_Model
import lib.Datasets.datasets as datasets
from lib.Models.initialization import WeightInit
from lib.Models.architectures import grow_classifier
from lib.cmdparser import parser
from lib.Training.train import train, train_gan
from lib.Training.validate import validate
from lib.Training.loss_functions import unified_loss_function as criterion
from lib.Utility.utils import save_checkpoint, save_task_checkpoint
from lib.Training.evaluate import get_latent_embedding
from lib.Utility.visualization import args_to_tensorboard
from lib.Utility.visualization import visualize_dataset_in_2d_embedding


# Comment this if CUDNN benchmarking is not desired
cudnn.benchmark = True


def main():
    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.debug:
        pdb.set_trace()

    if args.cross_dataset and not args.incremental_data:
        raise ValueError('cross-dataset training possible only if incremental-data flag set')

    # Check whether GPU is available and can be used
    # if CUDA is found then device is set accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Launch a writer for the tensorboard summary writer instance
    save_path = 'runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '_' + args.dataset + '_' + args.architecture +\
                '_variational_samples_' + str(args.var_samples) + '_latent_dim_' + str(args.var_latent_dim)

    # if we are resuming a previous training, note it in the name
    if args.resume:
        save_path = save_path + '_resumed'
    writer = SummaryWriter(save_path)

    # saving the parsed args to file
    log_file = os.path.join(save_path, "stdout")
    log = open(log_file, "a")
    for arg in vars(args):
        log.write(arg + ':' + str(getattr(args, arg)) + '\n')

    # Dataset loading
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    # get the number of classes from the class dictionary
    num_classes = dataset.num_classes

    # we set an epoch multiplier to 1 for isolated training and increase it proportional to amount of tasks in CL
    epoch_multiplier = 1
    # add command line options to TensorBoard
    args_to_tensorboard(writer, args)
    temp_embedding = []
    log.close()

    # Get a sample input from the data loader to infer color channels/size
    net_input, _ = next(iter(dataset.train_loader))
    # get the amount of color channels in the input images
    num_colors = net_input.size(1)

    # import model from architectures class
    net_init_method = getattr(architectures, args.architecture)

    # if we are not building an autoregressive model the number of output channels of the model is equivalent to
    # the amount of input channels. For an autoregressive models we set the number of output channels of the
    # non-autoregressive decoder portion according to the command line option below
    if not args.autoregression:
        args.out_channels = num_colors

    # build the model
    model = net_init_method(device, num_classes, num_colors, args)

    #optionally add the gan discriminator
    if args.gan:
        model.discriminator = Gan_Model(device, num_classes, num_colors, args)

    # Parallel container for multi GPU use and cast to available device
    # model = torch.nn.DataParallel(model).to(device)
    print(model)

    # Initialize the weights of the model, by default according to He et al.
    print("Initializing network with: " + args.weight_init)
    WeightInitializer = WeightInit(args.weight_init)
    WeightInitializer.init_model(model)

    # Define optimizer and loss function (criterion)
    optimizer = {}
    Encoder_param = list(model.encoder.parameters()) + list(model.latent_mu.parameters())+list(model.latent_std.parameters())+list(model.classifier.parameters()) +list(model.latent_decoder.parameters())+list(model.decoder.parameters())
    Gen_param = list(model.latent_decoder.parameters())+list(model.decoder.parameters())
    Dis_param = list(model.discriminator.parameters())
    optimizer['enc'] = torch.optim.Adam(Encoder_param, args.learning_rate)
    optimizer['gen'] = torch.optim.Adam(Gen_param, args.learning_rate_G)
    optimizer['dis'] = torch.optim.Adam(Dis_param, args.learning_rate_D)
    model = torch.nn.DataParallel(model).to(device)

    epoch = 0
    best_prec = 0
    best_loss = random.getrandbits(128)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # global encoder training
    while epoch < args.epochs:
        # if epoch+2 == epoch%args.epochs:
        #     print("debug perpose")
        # visualize the latent space before each task increment and at the end of training if it is 2-D
        if epoch % args.epochs == 0 and epoch > 0 or (epoch + 1) % (args.epochs * epoch_multiplier) == 0:
            if model.module.latent_dim == 2:
                print("Calculating and visualizing dataset embedding")
                # infer the number of current tasks to plot the different classes in the embedding
                num_tasks = num_classes

                zs = get_latent_embedding(model, dataset.train_loader, num_tasks, device)
                visualize_dataset_in_2d_embedding(writer, zs, args.dataset, save_path, task=num_tasks)

        # train
        train(dataset, model, criterion, epoch, optimizer['enc'], writer, device, args)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, writer, device, save_path, args)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        save_checkpoint({'epoch': epoch,
                         'arch': args.architecture,
                         'state_dict': model.state_dict(),
                         'best_prec': best_prec,
                         'best_loss': best_loss,
                         'optimizer': optimizer['enc'].state_dict()},
                        is_best, save_path)

        # increment epoch counters
        epoch += 1

    # Wgan-gp training
    best_prec = 0
    best_loss = random.getrandbits(128)
    while epoch < (args.gan_epochs):
        if epoch % args.epochs == 0 and epoch > 0 or (epoch + 1) % (args.epochs * epoch_multiplier) == 0:
            if model.module.latent_dim == 2:
                print("Calculating and visualizing dataset embedding")
                num_tasks = num_classes

                zs = get_latent_embedding(model, dataset.train_loader, num_tasks, device)
                visualize_dataset_in_2d_embedding(writer, zs, args.dataset, save_path, task=num_tasks)

        # train
        train_gan(dataset, model, criterion, epoch, optimizer, writer, device, args)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, writer, device, save_path, args)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        save_checkpoint({'epoch': epoch,
                         'arch': args.architecture,
                         'state_dict': model.state_dict(),
                         'best_prec': best_prec,
                         'best_loss': best_loss,
                         'optimizer': optimizer['gen'].state_dict()},
                        is_best, save_path)
        epoch += 1
    writer.close()


if __name__ == '__main__':     
    main()
