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
    if args.incremental_data:
        from lib.Datasets.incremental_dataset import get_incremental_dataset

        # get the method to create the incremental dataste (inherits from the chosen data loader)
        inc_dataset_init_method = get_incremental_dataset(data_init_method, args)

        # different options for class incremental vs. cross-dataset experiments
        if args.cross_dataset:
            # if a task order file is specified, load the task order from it
            if args.load_task_order:
                # check if file exists and if file ends with extension '.txt'
                if os.path.isfile(args.load_task_order) and len(args.load_task_order) >= 4\
                        and args.load_task_order[-4:] == '.txt':
                    print("=> loading task order from '{}'".format(args.load_task_order))
                    with open(args.load_task_order, 'rb') as fp:
                        task_order = pickle.load(fp)
                # if no file is found default to cmd line task order
                else:
                    # parse and split string at commas
                    task_order = args.dataset_order.split(',')
                    for i in range(len(task_order)):
                        # remove blank spaces in dataset names
                        task_order[i] = task_order[i].replace(" ", "")
            # use task order as specified in command line
            else:
                # parse and split string at commas
                task_order = args.dataset_order.split(',')
                for i in range(len(task_order)):
                    # remove blank spaces in dataset names
                    task_order[i] = task_order[i].replace(" ", "")

            # just for getting the number of classes in the first dataset
            num_classes = 0
            for i in range(args.num_base_tasks):
                temp_dataset_init_method = getattr(datasets, task_order[i])
                temp_dataset = temp_dataset_init_method(torch.cuda.is_available(), args)
                num_classes += temp_dataset.num_classes
                del temp_dataset

            # multiply epochs by number of tasks
            if args.num_increment_tasks:
                epoch_multiplier = ((len(task_order) - args.num_base_tasks) / args.num_increment_tasks) + 1
            else:
                # this branch will get active if num_increment_tasks is set to zero. This is useful when training
                # any isolated upper bound with all datasets present from the start.
                epoch_multiplier = 1.0
        else:
            # class incremental
            # if specified load task order from file
            if args.load_task_order:
                if os.path.isfile(args.load_task_order):
                    print("=> loading task order from '{}'".format(args.load_task_order))
                    task_order = np.load(args.load_task_order).tolist()
                else:
                    # if no file is found a random task order is created
                    print("=> no task order found. Creating randomized task order")
                    task_order = np.random.permutation(num_classes).tolist()
            else:
                # if randomize task order is specified create a random task order, else task order is sequential
                task_order = []
                for i in range(dataset.num_classes):
                    task_order.append(i)

                if args.randomize_task_order:
                    task_order = np.random.permutation(num_classes).tolist()

            # save the task order
            np.save(os.path.join(save_path, 'task_order.npy'), task_order)
            # set the number of classes to base tasks + 1 because base tasks is always one less.
            # E.g. if you have 2 classes it's one task. This is a little inconsistent from the naming point of view
            # but we wanted a single variable to work for both class incremental as well as cross-dataset experiments
            num_classes = args.num_base_tasks + 1
            # multiply epochs by number of tasks
            epoch_multiplier = ((len(task_order) - (args.num_base_tasks + 1)) / args.num_increment_tasks) + 1

        print("Task order: ", task_order)
        temp_embedding = []
        if args.wordvec:
            for cls_num in task_order:
                cls_name = list(dataset.class_to_idx.keys())[list(dataset.class_to_idx.values()).index(cls_num)]
                temp_embedding.append(dataset.wordvec[cls_name])             
            # dataset.wordvec = temp_embedding
        # log the task order into the text file
        log.write('task_order:' + str(task_order) + '\n')
        args.task_order = task_order

        # this is a little weird, but it needs to be here because the below method pops items from task_order
        args_to_tensorboard(writer, args)

        assert epoch_multiplier.is_integer(), print("uneven task division, make sure number of tasks are integers.")

        # Get the incremental dataset
        dataset = inc_dataset_init_method(torch.cuda.is_available(), device, task_order, args)
        if args.wordvec:
            dataset.wordvec_dict = copy.deepcopy(dataset.wordvec)
            dataset.wordvec = np.asarray(temp_embedding)
    else:
        # add command line options to TensorBoard
        args_to_tensorboard(writer, args)
        temp_embedding = []
        if args.wordvec:
            for cls_num in range(dataset.num_classes):
                cls_name = list(dataset.class_to_idx.keys())[cls_num]
                temp_embedding.append(dataset.wordvec[cls_name])
            dataset.wordvec_dict = copy.deepcopy(dataset.wordvec)
            dataset.wordvec = np.asarray(temp_embedding)

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
            # optimizer.load_state_dict(checkpoint['optimizer'])
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
                if args.incremental_data:
                    if args.cross_dataset:
                        num_tasks = sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                    else:
                        num_tasks = len(dataset.seen_tasks)

                zs = get_latent_embedding(model, dataset.train_loader, num_tasks, device)
                visualize_dataset_in_2d_embedding(writer, zs, args.dataset, save_path, task=num_tasks)

        # continual learning specific part
        if args.incremental_data:
            # at the end of each task increment
            if epoch % args.epochs == 0 and epoch > 0:
                print('Saving the last checkpoint from the previous task ...')
                save_task_checkpoint(save_path, epoch // args.epochs)

                print("Incrementing dataset ...")
                dataset.increment_tasks(model, args.batch_size, args.workers, writer, save_path,
                                        is_gpu=torch.cuda.is_available(),
                                        upper_bound_baseline=args.train_incremental_upper_bound,
                                        generative_replay=args.generative_replay,
                                        openset_generative_replay=args.openset_generative_replay,
                                        openset_threshold=args.openset_generative_replay_threshold,
                                        openset_tailsize=args.openset_weibull_tailsize,
                                        autoregression=args.autoregression)

                # grow the classifier and increment the variable for number of overall classes so we can use it later
                if args.cross_dataset:
                    grow_classifier(device, model.module.classifier,
                                    sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                                    - model.module.num_classes, WeightInitializer)
                    model.module.num_classes = sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                else:
                    # model.module.num_classes = 100
                    model.module.num_classes += args.num_increment_tasks
                    grow_classifier(device, model.module.classifier, args.num_increment_tasks, WeightInitializer)

                # reset moving averages etc. of the optimizer
                optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
                # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9, weight_decay = 0.00001)

            # change the number of seen classes
            if epoch % args.epochs == 0:
                model.module.seen_tasks = dataset.seen_tasks

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
    epoch = 0
    best_prec = 0
    best_loss = random.getrandbits(128)
    while epoch < (args.gan_epochs*epoch_multiplier):
        if epoch % args.epochs == 0 and epoch > 0 or (epoch + 1) % (args.epochs * epoch_multiplier) == 0:
            if model.module.latent_dim == 2:
                print("Calculating and visualizing dataset embedding")
                # infer the number of current tasks to plot the different classes in the embedding
                if args.incremental_data:
                    if args.cross_dataset:
                        num_tasks = sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                    else:
                        num_tasks = len(dataset.seen_tasks)

                zs = get_latent_embedding(model, dataset.train_loader, num_tasks, device)
                visualize_dataset_in_2d_embedding(writer, zs, args.dataset, save_path, task=num_tasks)

        # continual learning specific part
        if args.incremental_data:
            # at the end of each task increment
            if epoch % args.epochs == 0 and epoch > 0:
                print('Saving the last checkpoint from the previous task ...')
                save_task_checkpoint(save_path, epoch // args.epochs)

                print("Incrementing dataset ...")
                dataset.increment_tasks(model, args.batch_size, args.workers, writer, save_path,
                                        is_gpu=torch.cuda.is_available(),
                                        upper_bound_baseline=args.train_incremental_upper_bound,
                                        generative_replay=args.generative_replay,
                                        openset_generative_replay=args.openset_generative_replay,
                                        openset_threshold=args.openset_generative_replay_threshold,
                                        openset_tailsize=args.openset_weibull_tailsize,
                                        autoregression=args.autoregression)

                # grow the classifier and increment the variable for number of overall classes so we can use it later
                if args.cross_dataset:
                    grow_classifier(device, model.module.classifier,
                                    sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                                    - model.module.num_classes, WeightInitializer)
                    model.module.num_classes = sum(dataset.num_classes_per_task[:len(dataset.seen_tasks)])
                else:
                    # model.module.num_classes = 100
                    model.module.num_classes += args.num_increment_tasks
                    grow_classifier(device, model.module.classifier, args.num_increment_tasks, WeightInitializer)

                # reset moving averages etc. of the optimizer
                optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
                # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9, weight_decay = 0.00001)

            # change the number of seen classes
            if epoch % args.epochs == 0:
                model.module.seen_tasks = dataset.seen_tasks

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
