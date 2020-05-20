import torch
import torch.nn as nn


def unified_loss_function(output_samples_classification, target, output_samples_recon, inp, mu, std, device, args):
    """
    Computes the unified model's joint loss function consisting of a term for reconstruction, a KL term between
    approximate posterior and prior and the loss for the generative classifier. The number of variational samples
    is one per default, as specified in the command line parser and typically is how VAE models and also our unified
    model is trained. We have added the option to flexibly work with an arbitrary amount of samples.

    Parameters:
        output_samples_classification (torch.Tensor): Mini-batch of var_sample many classification prediction values.
        target (torch.Tensor): Classification targets for each element in the mini-batch.
        output_samples_recon (torch.Tensor): Mini-batch of var_sample many reconstructions.
        inp (torch.Tensor): The input mini-batch (before noise), aka the reconstruction loss' target.
        mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
        std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
        device (str): Device for computation.
        args (dict): Command line parameters. Needs to contain autoregression (bool).

    Returns:
        float: normalized classification loss
        float: normalized reconstruction loss
        float: normalized KL divergence
    """

    # for autoregressive models the decoder loss term corresponds to a classification based on 256 classes (for each
    # pixel value), i.e. a 256-way Softmax and thus a cross-entropy loss.
    # For regular decoders the loss is the reconstruction negative-log likelihood.
    if args.autoregression:
        recon_loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        # recon_loss = nn.BCEWithLogitsLoss(reduction='sum')
        recon_loss = nn.L1Loss(reduction='sum')
        # recon_loss = nn.L1Loss()

    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    recon_losses = torch.zeros(output_samples_recon.size(0)).to(device)
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # numerical value for stability of log computation
    eps = 1e-8

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)
        recon_losses[i] = recon_loss(output_samples_recon[i], inp) / torch.numel(inp)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)
    rl = torch.mean(recon_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)

    return cl, rl, kld
