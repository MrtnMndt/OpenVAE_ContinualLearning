"""
Stand alone evaluation script for open set recognition and plotting of different datasets

Uses the same command line parser as main.py

The attributes that need to be specified are the number of variational samples (should be greater than one if prediction
uncertainties are supposed to be calculated and compared), the architecture type and the resume flag pointing to a model
checkpoint file.
Other parameters like open set distance function etc. are optional.

Minimum example usage:
--resume /path/checkpoint.pth.tar --var-samples 100 -a MLP
"""

from lib.cmdparser import parser
import lib.Datasets.datasets as datasets
import lib.Models.architectures as architectures
from lib.Models.pixelcnn import PixelCNN
from lib.Training.evaluate import eval_dataset as eval_dataset
from lib.Training.evaluate import eval_openset_dataset as eval_openset_dataset
from lib.Utility.visualization import *
from lib.OpenSet.meta_recognition import *


def main():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Get the dataset which has been trained and the corresponding number of classes
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    num_classes = dataset.num_classes
    net_input, _ = next(iter(dataset.train_loader))
    num_colors = net_input.size(1)

    # Load open set dataset 1
    openset_data_init_method = getattr(datasets, args.openset_dataset)
    openset_dataset = openset_data_init_method(torch.cuda.is_available(), args)

    # Load open set dataset 2
    # Note: This could be easily refactored to one or flexible amount of datasets, we have kept this hard-coded
    # to reproduce the plots of the paper. Please feel free to refactor this.
    openset_data_init_method2 = getattr(datasets, args.openset_dataset2)
    openset_dataset2 = openset_data_init_method2(torch.cuda.is_available(), args)

    if not args.autoregression:
        args.out_channels = num_colors

    # Initialize empty model
    net_init_method = getattr(architectures, args.architecture)
    model = net_init_method(device, num_classes, num_colors, args)

    # Optional addition of autoregressive decoder portion
    if args.autoregression:
        model.pixelcnn = PixelCNN(device, num_colors, args.out_channels, args.pixel_cnn_channels,
                                  num_layers=args.pixel_cnn_layers, k=args.pixel_cnn_kernel_size,
                                  padding=args.pixel_cnn_kernel_size // 2)

    model = torch.nn.DataParallel(model).to(device)

    # load model (using the resume functionality)
    assert(os.path.isfile(args.resume)), "=> no model checkpoint found at '{}'".format(args.resume)

    # Fill the random model with the parameters of the checkpoint
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    best_prec = checkpoint['best_prec']
    best_loss = checkpoint['best_loss']
    # print the saved model's validation accuracy (as a check to see if the loaded model has really been trained)
    print("Saved model's validation accuracy: ", best_prec)
    print("Saved model's validation loss: ", best_loss)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # set the save path to the directory from which the model has been loaded
    save_path = os.path.dirname(args.resume)

    # start of the model evaluation on the training dataset and fitting
    print("Evaluating original train dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict_train = eval_dataset(model, dataset.train_loader, dataset.num_classes, device,
                                           samples=args.var_samples)
    print("Training accuracy: ", dataset_eval_dict_train["accuracy"])

    # Get the mean of z for correctly classified data inputs
    mean_zs = get_means(dataset_eval_dict_train["zs_correct"])

    # visualize the mean z vectors
    mean_zs_tensor = torch.stack(mean_zs, dim=0)
    visualize_means(mean_zs_tensor, dataset.class_to_idx, args.dataset, save_path, "z")

    # calculate each correctly classified example's distance to the mean z
    distances_to_z_means_correct_train = calc_distances_to_means(mean_zs, dataset_eval_dict_train["zs_correct"],
                                                                 args.distance_function)

    # Weibull fitting
    # set tailsize according to command line parameters (according to percentage of dataset size)
    tailsize = int(len(dataset.trainset) * args.openset_weibull_tailsize / num_classes)
    print("Fitting Weibull models with tailsize: " + str(tailsize))
    tailsizes = [tailsize] * num_classes
    weibull_models, valid_weibull = fit_weibull_models(distances_to_z_means_correct_train, tailsizes)
    assert valid_weibull, "Weibull fit is not valid"

    # ------------------------------------------------------------------------------------------
    # Fitting on train dataset complete. Beginning of all testing/open set recognition on validation and unknown sets.
    # ------------------------------------------------------------------------------------------
    # We evaluate the validation set to later evaluate trained dataset's statistical inlier/outlier estimates.
    print("Evaluating original validation dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict = eval_dataset(model, dataset.val_loader, dataset.num_classes, device, samples=args.var_samples)

    # Again calculate distances to mean z
    print("Validation accuracy: ", dataset_eval_dict["accuracy"])
    distances_to_z_means_correct = calc_distances_to_means(mean_zs, dataset_eval_dict["zs_correct"],
                                                           args.distance_function)

    # Evaluate outlier probability of trained dataset's validation set
    outlier_probs_correct = calc_outlier_probs(weibull_models, distances_to_z_means_correct)

    # Repeat process for open set recognition on unseen dataset 1 (
    print("Evaluating openset dataset: " + args.openset_dataset + ". This may take a while...")
    openset_dataset_eval_dict = eval_openset_dataset(model, openset_dataset.val_loader, openset_dataset.num_classes,
                                                     device, samples=args.var_samples)

    openset_distances_to_z_means = calc_distances_to_means(mean_zs, openset_dataset_eval_dict["zs"],
                                                           args.distance_function)

    openset_outlier_probs = calc_outlier_probs(weibull_models, openset_distances_to_z_means)

    # visualize the outlier probabilities
    visualize_weibull_outlier_probabilities(outlier_probs_correct, openset_outlier_probs,
                                            args.dataset, args.openset_dataset, save_path, tailsize)

    # getting outlier classification accuracies across the entire datasets
    dataset_classification_correct = calc_openset_classification(outlier_probs_correct, dataset.num_classes,
                                                                 num_outlier_threshs=100)
    openset_classification = calc_openset_classification(openset_outlier_probs, dataset.num_classes,
                                                         num_outlier_threshs=100)

    # open set recognition on unseen dataset 2 (Lots of redundant code copy pasting, could be refactored)
    print("Evaluating openset dataset 2: " + args.openset_dataset2 + ". This may take a while...")
    openset_dataset_eval_dict2 = eval_openset_dataset(model, openset_dataset2.val_loader, openset_dataset2.num_classes,
                                                      device, samples=args.var_samples)

    # joint prediction uncertainty plot for all datasets
    visualize_classification_uncertainty(dataset_eval_dict["out_mus_correct"],
                                         dataset_eval_dict["out_sigmas_correct"],
                                         openset_dataset_eval_dict["out_mus"],
                                         openset_dataset_eval_dict["out_sigmas"],
                                         openset_dataset_eval_dict2["out_mus"],
                                         openset_dataset_eval_dict2["out_sigmas"],
                                         args.dataset + ' (trained)', args.openset_dataset, args.openset_dataset2,
                                         args.var_samples, save_path)

    # get outlier probabilities of open set dataset 2
    openset_distances_to_z_means2 = calc_distances_to_means(mean_zs, openset_dataset_eval_dict2["zs"],
                                                            args.distance_function)

    openset_outlier_probs2 = calc_outlier_probs(weibull_models, openset_distances_to_z_means2)

    visualize_weibull_outlier_probabilities(outlier_probs_correct, openset_outlier_probs2,
                                            args.dataset, args.openset_dataset2, save_path, tailsize)

    # getting outlier classification accuracy for open set dataset 2
    openset_classification2 = calc_openset_classification(openset_outlier_probs2, dataset.num_classes,
                                                          num_outlier_threshs=100)

    # joint plot for outlier detection accuracy for seen and both unseen datasets
    visualize_openset_classification(dataset_classification_correct["outlier_percentage"],
                                     openset_classification["outlier_percentage"],
                                     openset_classification2["outlier_percentage"],
                                     args.dataset + ' (trained)', args.openset_dataset, args.openset_dataset2,
                                     dataset_classification_correct["thresholds"], save_path, tailsize)


if __name__ == '__main__':
    main()
