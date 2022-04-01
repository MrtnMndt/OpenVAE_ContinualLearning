import torch
import torch.nn as nn
import torch.nn.functional as F


def KLD(mu, std):
    # numerical value for stability of log computation
    eps = 1e-8

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)

    return kld


def joint_loss_function(output_samples_classification, target, output_samples_recon, inp, mu, std, device, args):
    """
    Computes the model's joint loss function consisting of a term for reconstruction, a KL term between
    approximate posterior and prior and the loss for the generative classifier. The number of samples
    is one per default, as specified in the command line parser and typically is how VAE models are trained.
    We have added the option to flexibly work with an arbitrary amount of samples.

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
        recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    recon_losses = torch.zeros(output_samples_recon.size(0)).to(device)
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)
        recon_losses[i] = recon_loss(output_samples_recon[i], inp) / torch.numel(inp)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)
    rl = torch.mean(recon_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = KLD(mu, std)

    return cl, rl, kld


def encoder_loss_with_fake(output_samples_classification, target, output_samples_recon, inp,
                           real_mu, real_std, rec_mu, rec_std, fake_mu, fake_std, device, args):
    """
    encoder loss for adversarial training of IntroVAE
    """
    recon_loss = nn.BCELoss(reduction='sum')

    class_loss = nn.CrossEntropyLoss(reduction='sum')

    cl = class_loss(output_samples_classification, target) / torch.numel(target)
    rl = recon_loss(output_samples_recon, inp) / torch.numel(inp)

    # Compute the KL divergences
    kld_real = KLD(real_mu, real_std)

    margin = args.margin / torch.numel(real_mu)
    kld_rec = F.relu(margin - KLD(rec_mu, rec_std))
    kld_fake = F.relu(margin - KLD(fake_mu, fake_std))

    return cl, rl, kld_real, kld_rec, kld_fake


def decoder_loss_with_fake(output_samples_recon, inp, rec_mu, rec_std, fake_mu, fake_std, args):
    """
    decoder loss for adversarial training of IntroVAE
    """
    recon_loss = nn.BCELoss(reduction='sum')

    rl = recon_loss(output_samples_recon, inp) / torch.numel(inp)

    kld_rec = KLD(rec_mu, rec_std)
    kld_fake = KLD(fake_mu, fake_std)

    return rl, kld_rec, kld_fake
