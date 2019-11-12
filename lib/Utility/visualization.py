import torch
import torchvision
import os
import math
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

# matplotlib backend, required for plotting of images to tensorboard
matplotlib.use('Agg')

# setting font sizes
title_font_size = 60
axes_font_size = 45
legend_font_size = 36
ticks_font_size = 48

# setting seaborn specifics
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
colors = sns.color_palette('Dark2', 7)
pal = sns.cubehelix_palette(10, light=0.0)


def args_to_tensorboard(writer, args):
    """
    Takes command line parser arguments and formats them to
    display them in TensorBoard text.

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        args (dict): dictionary of command line arguments
    """

    txt = ""
    for arg in vars(args):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"

    writer.add_text('command_line_parameters', txt, 0)


def visualize_image_grid(images, writer, count, name, save_path):
    """
    Visualizes a grid of images and saves it to both hard-drive as well as TensorBoard

    Parameters:
        images (torch.Tensor): Tensor of images.
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        count (int): counter usually specifying steps/epochs/time.
        name (str): name of the figure in tensorboard.
        save_path (str): path where image grid is going to be saved.
    """
    size = images.size(0)
    imgs = torchvision.utils.make_grid(images, nrow=int(math.sqrt(size)), padding=5)
    torchvision.utils.save_image(images, os.path.join(save_path, name + '_epoch_' + str(count + 1) + '.png'),
                                 nrow=int(math.sqrt(size)), padding=5)
    writer.add_image(name, imgs, count)


def visualize_confusion(writer, step, matrix, class_dict, save_path):
    """
    Visualization of confusion matrix. Is saved to hard-drive and TensorBoard.

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        step (int): Counter usually specifying steps/epochs/time.
        matrix (numpy.array): Square-shaped array of size class x class.
            Should specify cross-class accuracies/confusion in percent
            values (range 0-1).
        class_dict (dict): Dictionary specifying class names as keys and
            corresponding integer labels/targets as values.
        save_path (str): Path used for saving
    """

    all_categories = sorted(class_dict, key=class_dict.get)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Turn off the grid for this plot
    ax.grid(False)
    plt.tight_layout()

    writer.add_figure("Training data", fig, global_step=str(step))
    plt.savefig(os.path.join(save_path, 'confusion_epoch_' + str(step) + '.png'), bbox_inches='tight')


def visualize_dataset_in_2d_embedding(writer, encoding_list, dataset_name, save_path, task=1):
    """
    Visualization of 2-D latent embedding. Is saved to both hard-disc as well as TensorBoard.

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        encoding_list (list): List of Tensors containing encoding values
        dataset_name (str): Dataset name.
        save_path (str): Path used for saving.
        task (int): task counter. Used for naming.
    """

    num_classes = len(encoding_list)
    encoded_classes = []
    for i in range(len(encoding_list)):
        if isinstance(encoding_list[i], torch.Tensor):
            encoded_classes.append([i] * encoding_list[i].size(0))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoding_list[i] = torch.Tensor(encoding_list[i]).to(device)
            encoded_classes.append([i] * 0)
    encoded_classes = np.concatenate(np.asarray(encoded_classes), axis=0)
    encoding = torch.cat(encoding_list, dim=0)

    if encoding.size(1) != 2:
        print("Skipping visualization of latent space because it is not 2-D")
        return

    # select first and second dimension
    encoded_dim1 = np.squeeze(encoding.narrow(1, 0, 1).cpu().numpy())
    encoded_dim2 = np.squeeze(encoding.narrow(1, 1, 1).cpu().numpy())

    xlabel = 'z dimension 1'
    ylabel = 'z dimension 2'

    my_cmap = ListedColormap(sns.color_palette("Paired", num_classes).as_hex())
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(encoded_dim1, encoded_dim2, c=encoded_classes, cmap=my_cmap)

    plt.xlabel(xlabel, fontsize=axes_font_size)
    plt.ylabel(ylabel, fontsize=axes_font_size)
    plt.xticks(fontsize=ticks_font_size)
    plt.yticks(fontsize=ticks_font_size)

    cbar = plt.colorbar(ticks=np.linspace(0, num_classes-1, num_classes))
    cbar.ax.set_yticklabels([str(i) for i in range(num_classes)])
    cbar.ax.tick_params(labelsize=legend_font_size)

    plt.tight_layout()

    writer.add_figure('latent_embedding', fig, global_step=task)
    plt.savefig(os.path.join(save_path, dataset_name + '_latent_2d_embedding_task_' +
                             str(task) + '.png'), bbox_inches='tight')


def visualize_means(means, classes_order, data_name, save_path, name):
    """
    Visualization of means, e.g. of latent code z.

    Parameters:
        means (torch.Tensor): 2-D Tensor with one mean z vector per class.
        classes_order (dict): Defines mapping between integer indices and class names (strings).
        data_name (str): Dataset name. Used for naming.
        save_path (str): Saving path.
        name (str): Name for type of mean, e.g. "z".
    """
    classes_order = sorted(classes_order)
    classes = []
    for key in classes_order:
        classes.append(key)

    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(means.cpu().numpy(), cmap="BrBG")
    ax.set_title(data_name, fontsize=title_font_size)
    ax.set_xlabel(name + ' mean activations', fontsize=axes_font_size)
    ax.set_yticklabels(classes, rotation=0)
    plt.savefig(os.path.join(save_path, name + '_mean_activations.png'), bbox_inches='tight')


def visualize_classification_uncertainty(data_mus, data_sigmas, other_data_mus, other_data_sigmas, other_data_mus2,
                                         other_data_sigmas2, data_name, other_data_name, other_data_name2,
                                         num_samples, save_path):
    """
    Visualization of prediction uncertainty computed over multiple samples for each input.

    Could be refactored for flexible amount of other_data mus and sigmas and corresponding dataset names. Currently
    hard-coded to two to reproduce the plots of the paper.

    Parameters:
        data_mus (list or torch.Tensor): Encoded mu values for trained dataset's validation set.
        data_sigmas (list or torch.Tensor): Encoded sigma values for trained dataset's validation set.
        other_data_mus (list or torch.Tensor): Encoded mu values for unseen dataset.
        other_data_sigmas (list or torch.Tensor): Encoded sigma values for unseen dataset.
        other_data_mus2 (list or torch.Tensor): Encoded mu values for second unseen dataset.
        other_data_sigmas2 (list or torch.Tensor): Encoded sigma values for second unseen dataset.
        data_name (str): Original dataset's name.
        other_data_name (str): Name of first unseen dataset.
        other_data_name2 (str): Name of second unseen dataset.
        num_samples (int): Number of used samples to obtain prediction values.
        save_path (str): Saving path.
    """

    data_mus = [y for x in data_mus for y in x]
    data_sigmas = [y for x in data_sigmas for y in x]
    other_data_mus = [y for x in other_data_mus for y in x]
    other_data_sigmas = [y for x in other_data_sigmas for y in x]

    other_data_mus2 = [y for x in other_data_mus2 for y in x]
    other_data_sigmas2 = [y for x in other_data_sigmas2 for y in x]

    plt.figure(figsize=(20, 14))
    plt.scatter(data_mus, data_sigmas, label=data_name, s=75, c=colors[2], alpha=1.0)
    plt.scatter(other_data_mus, other_data_sigmas, label=other_data_name, s=75, c=colors[1], alpha=0.3,
                marker='*')
    plt.scatter(other_data_mus2, other_data_sigmas2, label=other_data_name2, s=75, c=colors[4], alpha=0.3,
                marker='*')
    plt.xlabel("Prediction mean", fontsize=axes_font_size)
    plt.ylabel("Prediction standard deviation", fontsize=axes_font_size)
    plt.xlim(xmin=-0.05, xmax=1.05)
    plt.ylim(ymin=-0.05, ymax=0.55)
    plt.legend(loc=1, fontsize=legend_font_size)
    plt.savefig(os.path.join(save_path, data_name + '_vs_' + other_data_name + '_' + other_data_name2 +
                             '_classification_uncertainty_' + str(num_samples) + '_samples.pdf'),
                bbox_inches='tight')


def visualize_weibull_outlier_probabilities(data_outlier_probs, other_data_outlier_probs,
                                            data_name, other_data_name, save_path, tailsize):
    """
    Visualization of Weibull CDF outlier probabilites.

    Parameters:
        data_outlier_probs (np.array): Outlier probabilities for each input of the trained dataset's validation set.
        other_data_outlier_probs (np.array): Outlier probabilities for each input of an unseen dataset.
        data_name (str): Original trained dataset's name.
        other_data_name (str): Unseen dataset's name.
        save_path (str): Saving path.
        tailsize (int): Fitted Weibull model's tailsize.
    """

    data_outlier_probs = np.concatenate(data_outlier_probs, axis=0)
    other_data_outlier_probs = np.concatenate(other_data_outlier_probs, axis=0)

    data_weights = np.ones_like(data_outlier_probs) / float(len(data_outlier_probs))
    other_data_weights = np.ones_like(other_data_outlier_probs) / float(len(other_data_outlier_probs))

    plt.figure(figsize=(20, 20))
    plt.hist(data_outlier_probs, label=data_name, weights=data_weights, bins=50, color=colors[0],
             alpha=0.5, edgecolor='white', linewidth=5)
    plt.hist(other_data_outlier_probs, label=other_data_name, weights=other_data_weights,
             bins=50, color=colors[1], alpha=0.5, edgecolor='white', linewidth=5)
    plt.title("Outlier probabilities: tailsize " + str(tailsize), fontsize=title_font_size)
    plt.xlabel("Outlier probability according to Weibull CDF", fontsize=axes_font_size)
    plt.ylabel("Percentage", fontsize=axes_font_size)
    plt.xlim(xmin=-0.05, xmax=1.05)
    plt.ylim(ymin=-0.05, ymax=1.05)
    plt.legend(loc=0)
    plt.savefig(os.path.join(save_path, data_name + '_' + other_data_name + '_weibull_outlier_probabilities_tailsize_'
                             + str(tailsize) + '.png'), bbox_inches='tight')


def visualize_openset_classification(data, other_data, other_data2, data_name, other_data_name, other_data_name2,
                                     thresholds, save_path, tailsize):
    """
    Visualization of percentage of datasets considered as statistical outliers evaluated for different
    Weibull CDF rejection priors.

    Could be refactored for flexible amount of other datasets and corresponding dataset names. Currently
    hard-coded to two to reproduce the plots of the paper.

    Parameters:
        data (list): Dataset outlier percentages per rejection prior value for the trained dataset's validation set.
        other_data (list): Dataset outlier percentages per rejection prior value for an unseen dataset.
        other_data2 (list): Dataset outlier percentages per rejection prior value for a second unseen dataset.
        data_name (str): Original trained dataset's name.
        other_data_name (str): First unseen dataset's name.
        other_data_name2 (str): Second unseen dataset's name.
        thresholds (list): List of integers with rejection prior values.
        save_path (str): Saving path.
        tailsize (int): Weibull model's tailsize.
    """

    lw = 8
    plt.figure(figsize=(20, 14))
    plt.plot(thresholds, data, label=data_name, color=colors[2], linestyle='solid', linewidth=lw)
    plt.plot(thresholds, other_data, label=other_data_name, color=colors[1], linestyle='dashed', linewidth=lw)
    plt.plot(thresholds, other_data2, label=other_data_name2, color=colors[4], linestyle='-.', linewidth=lw)
    plt.xlabel(r"Weibull CDF outlier rejection prior $\Omega_t$", fontsize=axes_font_size)
    plt.ylabel("Percentage of dataset outliers", fontsize=axes_font_size)
    plt.xlim(xmin=-0.05, xmax=1.05)
    plt.ylim(ymin=-0.05, ymax=1.05)
    plt.legend(loc=0, fontsize=legend_font_size)
    plt.savefig(os.path.join(save_path, data_name + '_' + other_data_name + '_' + other_data_name2 +
                             '_outlier_classification' + '_tailsize_' + str(tailsize) + '.pdf'),
                bbox_inches='tight')
