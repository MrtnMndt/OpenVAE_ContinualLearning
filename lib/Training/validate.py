import time
import math
import torch
import torch.nn.functional as F
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import ConfusionMeter
from lib.Utility.metrics import accuracy
from lib.Utility.visualization import visualize_confusion
from lib.Utility.visualization import visualize_image_grid


def validate(Dataset, model, criterion, epoch, writer, device, save_path, args):
    """
    Evaluates/validates the model

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        epoch (int): Epoch counter
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        save_path (str): path to save data to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), epochs (int), incremental_data (bool), autoregression (bool),
            visualization_epoch (int), num_base_tasks (int), num_increment_tasks (int) and
            patch_size (int).

    Returns:
        float: top1 precision/accuracy
        float: average loss
    """

    # initialize average meters to accumulate values
    class_losses = AverageMeter()
    recon_losses_nat = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()

    # for autoregressive models add an additional instance for reconstruction loss in bits per dimension
    if args.autoregression:
        recon_losses_bits_per_dim = AverageMeter()

    # for continual learning settings also add instances for base and new reconstruction metrics
    # corresponding accuracy values are calculated directly from the confusion matrix below
    if args.incremental_data and ((epoch + 1) % args.epochs == 0 and epoch > 0):
        recon_losses_new_nat = AverageMeter()
        recon_losses_base_nat = AverageMeter()
        if args.autoregression:
            recon_losses_new_bits_per_dim = AverageMeter()
            recon_losses_base_bits_per_dim = AverageMeter()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    # confusion matrix
    confusion = ConfusionMeter(model.module.num_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # evaluate the entire validation dataset
    with torch.no_grad():
        for i, (inp, target) in enumerate(Dataset.val_loader):
            inp = inp.to(device)
            target = target.to(device)

            recon_target = inp
            class_target = target

            # compute output
            class_samples, recon_samples, mu, std = model(inp)

            # for autoregressive models convert the target to 0-255 integers and compute the autoregressive decoder
            # for each sample
            if args.autoregression:
                recon_target = (recon_target * 255).long()
                recon_samples_autoregression = torch.zeros(recon_samples.size(0), inp.size(0), 256, inp.size(1),
                                                           inp.size(2), inp.size(3)).to(device)
                for j in range(model.module.num_samples):
                    recon_samples_autoregression[j] = model.module.pixelcnn(
                        inp, torch.sigmoid(recon_samples[j])).contiguous()
                recon_samples = recon_samples_autoregression

            # compute loss
            class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu,
                                                         std, device, args)

            # For autoregressive models also update the bits per dimension value, converted from the obtained nats
            if args.autoregression:
                recon_losses_bits_per_dim.update(recon_loss.item() * math.log2(math.e), inp.size(0))

            # take mean to compute accuracy
            # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
            class_output = torch.mean(class_samples, dim=0)
            recon_output = torch.mean(recon_samples, dim=0)

            # measure accuracy, record loss, fill confusion matrix
            prec1 = accuracy(class_output, target)[0]
            top1.update(prec1.item(), inp.size(0))
            confusion.add(class_output.data, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # for autoregressive models generate reconstructions by sequential sampling from the
            # multinomial distribution (Reminder: the original output is a 255 way Softmax as PixelVAEs are posed as a
            # classification problem). This serves two purposes: visualization of reconstructions and computation of
            # a reconstruction loss in nats using a BCE loss, comparable to that of a regular VAE.
            recon_target = inp
            if args.autoregression:
                recon = torch.zeros((inp.size(0), inp.size(1), inp.size(2), inp.size(3))).to(device)
                for h in range(inp.size(2)):
                    for w in range(inp.size(3)):
                        for c in range(inp.size(1)):
                            probs = torch.softmax(recon_output[:, :, c, h, w], dim=1).data
                            pixel_sample = torch.multinomial(probs, 1).float() / 255.
                            recon[:, c, h, w] = pixel_sample.squeeze()

                if (epoch % args.visualization_epoch == 0) and (i == (len(Dataset.val_loader) - 1)) and (epoch > 0):
                    visualize_image_grid(recon, writer, epoch + 1, 'reconstruction_snapshot', save_path)

                recon_loss = F.binary_cross_entropy(recon, recon_target)
            else:
                # If not autoregressive simply apply the Sigmoid and visualize
                recon = torch.sigmoid(recon_output)
                if (i == (len(Dataset.val_loader) - 1)) and (epoch % args.visualization_epoch == 0) and (epoch > 0):
                    visualize_image_grid(recon, writer, epoch + 1, 'reconstruction_snapshot', save_path)

            # update the respective loss values. To be consistent with values reported in the literature we scale
            # our normalized losses back to un-normalized values.
            # For the KLD this also means the reported loss is not scaled by beta, to allow for a fair comparison
            # across potential weighting terms.
            class_losses.update(class_loss.item() * model.module.num_classes, inp.size(0))
            kld_losses.update(kld_loss.item() * model.module.latent_dim, inp.size(0))
            recon_losses_nat.update(recon_loss.item() * inp.size()[1:].numel(), inp.size(0))
            losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))

            # if we are learning continually, we need to calculate the base and new reconstruction losses at the end
            # of each task increment.
            if args.incremental_data and ((epoch + 1) % args.epochs == 0 and epoch > 0):
                for j in range(inp.size(0)):
                    # get the number of classes for class incremental scenarios.
                    base_classes = model.module.seen_tasks[:args.num_base_tasks + 1]
                    new_classes = model.module.seen_tasks[-args.num_increment_tasks:]

                    if args.autoregression:
                        rec = recon_output[j].view(1, recon_output.size(1), recon_output.size(2),
                                                   recon_output.size(3), recon_output.size(4))
                        rec_tar = recon_target[j].view(1, recon_target.size(1), recon_target.size(2),
                                                       recon_target.size(3))

                    # If the input belongs to one of the base classes also update base metrics
                    if class_target[j].item() in base_classes:
                        if args.autoregression:
                            recon_losses_base_bits_per_dim.update(F.cross_entropy(rec, (rec_tar * 255).long()) *
                                                                  math.log2(math.e), 1)
                        recon_losses_base_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)
                    # if the input belongs to one of the new classes also update new metrics
                    elif class_target[j].item() in new_classes:
                        if args.autoregression:
                            recon_losses_new_bits_per_dim.update(F.cross_entropy(rec, (rec_tar * 255).long()) *
                                                                 math.log2(math.e), 1)
                        recon_losses_new_nat.update(F.binary_cross_entropy(recon[j], recon_target[j]), 1)

            # If we are at the end of validation, create one mini-batch of example generations. Only do this every
            # other epoch specified by visualization_epoch to avoid generation of lots of images and computationally
            # expensive calculations of the autoregressive model's generation.
            if i == (len(Dataset.val_loader) - 1) and epoch % args.visualization_epoch == 0 and (epoch > 0):
                # generation
                gen = model.module.generate()

                if args.autoregression:
                    gen = model.module.pixelcnn.generate(gen)
                visualize_image_grid(gen, writer, epoch + 1, 'generation_snapshot', save_path)

            # Print progress
            if i % args.print_freq == 0:
                print('Validate: [{0}][{1}/{2}]\t' 
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                      'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                       epoch+1, i, len(Dataset.val_loader), batch_time=batch_time, loss=losses, cl_loss=class_losses,
                       top1=top1, recon_loss=recon_losses_nat, KLD_loss=kld_losses))

    # TensorBoard summary logging
    writer.add_scalar('validation/val_precision@1', top1.avg, epoch)
    writer.add_scalar('validation/val_average_loss', losses.avg, epoch)
    writer.add_scalar('validation/val_class_loss', class_losses.avg, epoch)
    writer.add_scalar('validation/val_recon_loss_nat', recon_losses_nat.avg, epoch)
    writer.add_scalar('validation/val_KLD', kld_losses.avg, epoch)

    if args.autoregression:
        writer.add_scalar('validation/val_recon_loss_bits_per_dim', recon_losses_bits_per_dim.avg, epoch)

    print(' * Validation: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

    # At the end of training isolated, or at the end of every task visualize the confusion matrix
    if (epoch + 1) % args.epochs == 0 and epoch > 0:
        # visualize the confusion matrix
        visualize_confusion(writer, epoch + 1, confusion.value(), Dataset.class_to_idx, save_path)

        # If we are in a continual learning scenario, also use the confusion matrix to extract base and new precision.
        if args.incremental_data:
            prec1_base = 0.0
            prec1_new = 0.0
            # this has to be + 1 because the number of initial tasks is always one less than the amount of classes
            # i.e. 1 task is 2 classes etc.
            for c in range(args.num_base_tasks + 1):
                prec1_base += confusion.value()[c][c]
            prec1_base = prec1_base / (args.num_base_tasks + 1)

            # For the first task "new" metrics are equivalent to "base"
            if (epoch + 1) / args.epochs == 1:
                prec1_new = prec1_base
                recon_losses_new_nat.avg = recon_losses_base_nat.avg
                if args.autoregression:
                    recon_losses_new_bits_per_dim.avg = recon_losses_base_bits_per_dim.avg
            else:
                for c in range(args.num_increment_tasks):
                    prec1_new += confusion.value()[-c-1][-c-1]
                prec1_new = prec1_new / args.num_increment_tasks

            # At the continual learning metrics to TensorBoard
            writer.add_scalar('validation/base_precision@1', prec1_base, len(model.module.seen_tasks)-1)
            writer.add_scalar('validation/new_precision@1', prec1_new, len(model.module.seen_tasks)-1)
            writer.add_scalar('validation/base_rec_loss_nats', recon_losses_base_nat.avg * args.patch_size *
                              args.patch_size * model.module.num_colors, len(model.module.seen_tasks) - 1)
            writer.add_scalar('validation/new_rec_loss_nats', recon_losses_new_nat.avg * args.patch_size *
                              args.patch_size * model.module.num_colors, len(model.module.seen_tasks) - 1)

            if args.autoregression:
                writer.add_scalar('validation/base_rec_loss_bits_per_dim',
                                  recon_losses_base_bits_per_dim.avg, len(model.module.seen_tasks) - 1)
                writer.add_scalar('validation/new_rec_loss_bits_per_dim',
                                  recon_losses_new_bits_per_dim.avg, len(model.module.seen_tasks) - 1)

            print(' * Incremental validation: Base Prec@1 {prec1_base:.3f} New Prec@1 {prec1_new:.3f}\t'
                  'Base Recon Loss {recon_losses_base_nat.avg:.3f} New Recon Loss {recon_losses_new_nat.avg:.3f}'
                  .format(prec1_base=100*prec1_base, prec1_new=100*prec1_new,
                          recon_losses_base_nat=recon_losses_base_nat, recon_losses_new_nat=recon_losses_new_nat))

    return top1.avg, losses.avg
