########################
# Importing libraries
########################
# System libraries
import os
import random
from time import gmtime, strftime
import numpy as np

# Tensorboard for PyTorch logging and visualization
from torch.utils.tensorboard import SummaryWriter

# Torch libraries
import torch
import torch.backends.cudnn as cudnn

# Custom library
import lib.Models.architectures as architectures
from lib.Models.pixelcnn import PixelCNN
import lib.Datasets.datasets as datasets
from lib.Models.initialization import WeightInit
from lib.Models.architectures import grow_classifier
from lib.cmdparser import parser
from lib.Training.train import train
from lib.Training.validate import validate
from lib.Training.loss_functions import joint_loss_function as criterion
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

    # Check whether GPU is available and can be used
    # if CUDA is found then device is set accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Launch a writer for the tensorboard summary writer instance
    save_path = 'runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '_' + args.dataset + '_' + args.architecture +\
                '_variational_samples_' + str(args.var_samples) + '_latent_dim_' + str(args.var_latent_dim)

    # add option specific naming to separate tensorboard log files later
    if args.autoregression:
        save_path += '_pixelvae'
    if args.introspection:
        save_path += '_introvae'

    if args.autoregression and args.introspection:
        raise ValueError('Autoregressive model variant with introspection not implemented')

    if args.incremental_data:
        save_path += '_incremental'
        if args.train_incremental_upper_bound:
            save_path += '_upper_bound'
        if args.generative_replay:
            save_path += '_genreplay'
        if args.openset_generative_replay:
            save_path += '_opensetreplay'

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
        inc_dataset_init_method = get_incremental_dataset(data_init_method)

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
        # log the task order into the text file
        log.write('task_order:' + str(task_order) + '\n')
        args.task_order = task_order

        # this is a little weird, but it needs to be here because the below method pops items from task_order
        args_to_tensorboard(writer, args)

        assert epoch_multiplier.is_integer(), print("uneven task division, make sure number of tasks are integers.")

        # Get the incremental dataset
        dataset = inc_dataset_init_method(torch.cuda.is_available(), device, task_order, args)
    else:
        # add command line options to TensorBoard
        args_to_tensorboard(writer, args)

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

    # optionally add the autoregressive decoder
    if args.autoregression:
        model.pixelcnn = PixelCNN(device, num_colors, args.out_channels, args.pixel_cnn_channels,
                                  num_layers=args.pixel_cnn_layers, k=args.pixel_cnn_kernel_size,
                                  padding=args.pixel_cnn_kernel_size//2)

    # Parallel container for multi GPU use and cast to available device
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    # Initialize the weights of the model, by default according to He et al.
    print("Initializing network with: " + args.weight_init)
    WeightInitializer = WeightInit(args.weight_init)
    WeightInitializer.init_model(model)

    # Define optimizer and loss function (criterion)
    if args.introspection:
        from lib.Training.loss_functions import encoder_loss_with_fake as criterion_enc
        from lib.Training.loss_functions import decoder_loss_with_fake as criterion_dec
        train_criterion = [criterion_enc, criterion_dec]

        # Define optimizer and loss function (criterion)
        encoder_param = list(model.module.encoder.parameters()) + list(model.module.latent_mu.parameters()) + list(
            model.module.latent_std.parameters()) + list(model.module.classifier.parameters())
        if args.architecture == 'MLP':
            gen_param = list(model.module.decoder.parameters())
        else:
            gen_param = list(model.module.latent_decoder.parameters()) + list(model.module.decoder.parameters())
        optimizer_enc = torch.optim.Adam(encoder_param, args.learning_rate)
        optimizer_dec = torch.optim.Adam(gen_param, args.learning_rate)
        optimizer = [optimizer_enc, optimizer_dec]
    else:
        # just use the optimizer and the full imported loss function
        train_criterion = criterion
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

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
            if args.introspection:
                optimizer[0].load_state_dict(checkpoint['optimizer_enc'])
                optimizer[1].load_state_dict(checkpoint['optimizer_dec'])
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optimize until final amount of epochs is reached. Final amount of epochs is determined through the
    while epoch < (args.epochs * epoch_multiplier):
        # visualize the latent space before each task increment and at the end of training if it is 2-D
        if epoch % args.epochs == 0 and epoch > 0 or (epoch + 1) % (args.epochs * epoch_multiplier) == 0:
            if model.module.latent_dim == 2:
                print("Calculating and visualizing dataset embedding")
                # infer the number of current tasks to plot the different classes in the embedding
                if args.incremental_data:
                    num_tasks = len(dataset.seen_tasks)
                else:
                    num_tasks = num_classes

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
                model.module.num_classes += args.num_increment_tasks
                grow_classifier(device, model.module.classifier, args.num_increment_tasks, WeightInitializer)

                # reset moving averages etc. of the optimizer
                if args.introspection:
                    # Define optimizer and loss function (criterion)
                    encoder_param = list(model.module.encoder.parameters()) + list(
                        model.module.latent_mu.parameters()) + list(
                        model.module.latent_std.parameters()) + list(model.module.classifier.parameters())
                    if args.architecture == 'MLP':
                        gen_param = list(model.module.decoder.parameters())
                    else:
                        gen_param = list(model.module.latent_decoder.parameters()) + list(
                            model.module.decoder.parameters())
                    optimizer_enc = torch.optim.Adam(encoder_param, args.learning_rate)
                    optimizer_dec = torch.optim.Adam(gen_param, args.learning_rate)
                    optimizer = [optimizer_enc, optimizer_dec]
                else:
                    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

            # change the number of seen classes
            if epoch % args.epochs == 0:
                model.module.seen_tasks = dataset.seen_tasks

        # train
        train(dataset, model, train_criterion, epoch, optimizer, writer, device, args)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, writer, device, save_path, args)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        if args.introspection:
            save_checkpoint({'epoch': epoch,
                             'arch': args.architecture,
                             'state_dict': model.state_dict(),
                             'best_prec': best_prec,
                             'best_loss': best_loss,
                             'optimizer_enc': optimizer[0].state_dict(),
                             'optimizer_dec': optimizer[1].state_dict()},
                            is_best, save_path)
        else:
            save_checkpoint({'epoch': epoch,
                             'arch': args.architecture,
                             'state_dict': model.state_dict(),
                             'best_prec': best_prec,
                             'best_loss': best_loss,
                             'optimizer': optimizer.state_dict()},
                            is_best, save_path)

        # increment epoch counters
        epoch += 1

        # if a new task begins reset the best prec so that new best model can be stored.
        if args.incremental_data and epoch % args.epochs == 0:
            best_prec = 0
            best_loss = random.getrandbits(128)

    writer.close()


if __name__ == '__main__':     
    main()
