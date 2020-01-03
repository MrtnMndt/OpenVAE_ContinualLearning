import time
import torch
import torch.nn as nn
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy


def train(Dataset, model, criterion, epoch, optimizer, writer, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and log_weights (bool).
    """

    # Create instances to accumulate losses etc.
    class_losses = AverageMeter()
    recon_losses = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    wordvec=[]
    if args.wordvec:
        wordvec = torch.from_numpy(Dataset.wordvec).float().to(device)
    # train
    for i, (inp, target) in enumerate(Dataset.train_loader):
        inp = inp.to(device)
        target = target.to(device)


        recon_target = inp
        class_target = target

        # this needs to be below the line where the reconstruction target is set
        # sample and add noise to the input (but not to the target!).
        if args.denoising_noise_value > 0.0:
            noise = torch.randn(inp.size()).to(device) * args.denoising_noise_value
            inp = inp + noise

        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        class_samples, recon_samples, mu, std = model(inp)

        # if we have an autoregressive model variant, further calculate the corresponding layers.
        if args.autoregression:
            recon_samples_autoregression = torch.zeros(recon_samples.size(0), inp.size(0), 256, inp.size(1),
                                                       inp.size(2), inp.size(3)).to(device)
            for j in range(model.module.num_samples):
                recon_samples_autoregression[j] = model.module.pixelcnn(recon_target,
                                                                        torch.sigmoid(recon_samples[j])).contiguous()
            recon_samples = recon_samples_autoregression
            # set the target to work with the 256-way Softmax
            recon_target = (recon_target * 255).long()

        # calculate loss
        mu_c = None
        if args.wordvec:
            # word_embedding = model.module.word_embedding(wordvec[:model.module.num_classes])
            word_embedding = model.module.word_embedding(wordvec)
            mu_c = word_embedding[target]

        class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, mu_c, std,
                                                     device, args)

        # add the individual loss components together and weight the KL term.
        re_weight = 1
        if args.no_recon:
            re_weight = 0 

        loss = class_loss + re_weight * recon_loss + args.var_beta * kld_loss
        if args.distill_wordvec:
            word_embedding_class = model.module.classifier(word_embedding)
            w_class_loss = nn.CrossEntropyLoss()(word_embedding_class, torch.arange(model.module.num_classes).to(device))
            # w_class_loss = nn.CrossEntropyLoss()(word_embedding_class, torch.arange(100).to(device))
            loss +=w_class_loss

        # take mean to compute accuracy. Note if variational samples are 1 this only gets rid of a dummy dimension.
        output = torch.mean(class_samples, dim=0)

        # record precision/accuracy and losses
        prec1 = accuracy(output, target)[0]
        top1.update(prec1.item(), inp.size(0))
        losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))
        class_losses.update(class_loss.item(), inp.size(0))
        recon_losses.update(recon_loss.item(), inp.size(0))
        kld_losses.update(kld_loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if i % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                  'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                   epoch+1, i, len(Dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, cl_loss=class_losses, top1=top1,
                   recon_loss=recon_losses, KLD_loss=kld_losses))

    # TensorBoard summary logging
    writer.add_scalar('training/train_precision@1', top1.avg, epoch)
    writer.add_scalar('training/train_average_loss', losses.avg, epoch)
    writer.add_scalar('training/train_KLD', kld_losses.avg, epoch)
    writer.add_scalar('training/train_class_loss', class_losses.avg, epoch)
    writer.add_scalar('training/train_recon_loss', recon_losses.avg, epoch)
    if args.distill_wordvec:
        writer.add_scalar('training/train_w_class_loss', w_class_loss.item(), epoch)

    # If the log weights argument is specified also add parameter and gradient histograms to TensorBoard.
    if args.log_weights:
        # Histograms and distributions of network parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins="auto")
            # second check required for buffers that appear in the parameters dict but don't receive gradients
            if value.requires_grad and value.grad is not None:
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins="auto")

    print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))


def train_gan(Dataset, model, criterion, epoch, optimizer, writer, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and log_weights (bool).
    """

    # Create instances to accumulate losses etc.
    class_losses = AverageMeter()
    recon_losses = AverageMeter()
    Gen_losses = AverageMeter()
    Dis_losses = AverageMeter()
    Real_losses = AverageMeter()
    Fake_losses = AverageMeter()
    losses = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    wordvec=[]
    if args.wordvec:
        wordvec = torch.from_numpy(Dataset.wordvec).float().to(device)
    # train
    for i, (inp, target) in enumerate(Dataset.train_loader):
        inp = inp.to(device)
        target = target.to(device)

        recon_target = inp
        class_target = target

        # this needs to be below the line where the reconstruction target is set
        # sample and add noise to the input (but not to the target!).
        if args.denoising_noise_value > 0.0:
            noise = torch.randn(inp.size()).to(device) * args.denoising_noise_value
            inp = inp + noise

        # measure data loading time
        data_time.update(time.time() - end)

        #D
        if i % 2 ==0:
            optimizer['dis'].zero_grad()
            D_real = model.module.discriminator(inp)
            D_real_loss = -torch.mean(D_real)

            class_samples, recon_samples, mu, std = model(inp)
            v,b,c,x,y = recon_samples.shape
            recon_samples= recon_samples.view(v*b,c,x,y)
            D_fake = model.module.discriminator(recon_samples)
            D_fake_loss = torch.mean(D_fake)

            #gradient penalty
            alpha = torch.rand((inp.shape[0],1,1,1)).to(device)
            x_hat = (alpha * inp.data + (1-alpha) * recon_samples.data).requires_grad_(True)
            pred_x_hat = model.module.discriminator(x_hat)
            D_gp_loss = model.module.discriminator.gradient_penalty(pred_x_hat, x_hat)
            D_loss = D_real_loss + D_fake_loss + D_gp_loss
            Dis_losses.update(D_loss.item())
            Real_losses.update(D_real_loss.item())

            optimizer['dis'].zero_grad()
            optimizer['gen'].zero_grad()
            D_loss.backward()
            optimizer['dis'].step()

        # if i % 5 ==0:
        #G
        class_samples, recon_samples, mu, std = model(inp)
        v,b,c,x,y = recon_samples.shape
        recon_samples= recon_samples.view(v*b,c,x,y)
        D_fake = model.module.discriminator(recon_samples)
        recon_loss = nn.L1Loss()(recon_samples,inp)
        G_fake_loss = -torch.mean(D_fake)
        G_loss =  G_fake_loss #+ recon_loss
        Gen_losses.update(G_loss.item())
        Fake_losses.update((D_fake_loss+G_fake_loss).item())

        optimizer['dis'].zero_grad()
        optimizer['gen'].zero_grad()
        G_loss.backward()
        optimizer['gen'].step()

        # take mean to compute accuracy. Note if variational samples are 1 this only gets rid of a dummy dimension.
        output = torch.mean(class_samples, dim=0)

        # record precision/accuracy and losses
        prec1 = accuracy(output, target)[0]
        top1.update(prec1.item(), inp.size(0))

        class_loss = nn.CrossEntropyLoss()(output, target)
        recon_loss = nn.MSELoss()(recon_samples,inp)

        class_losses.update(class_loss.item(), inp.size(0))
        recon_losses.update(recon_loss.item(), inp.size(0))
        losses.update((G_loss+D_loss).item(), inp.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if i % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'D Loss {D_loss.val:.4f} ({D_loss.avg:.4f})\t'
                  'G Loss {G_loss.val:.4f} ({G_loss.avg:.4f})\t'
                  'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'.format(
                   epoch+1, i, len(Dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, D_loss=Dis_losses, G_loss=Gen_losses, cl_loss=class_losses, top1=top1,
                   recon_loss=recon_losses))

    # TensorBoard summary logging
    writer.add_scalar('training/train_precision@1', top1.avg, epoch)
    writer.add_scalar('training/train_class_loss', class_losses.avg, epoch)
    writer.add_scalar('training/train_recon_loss', recon_losses.avg, epoch)
    writer.add_scalar('training/train_average_loss', losses.avg, epoch)
    writer.add_scalar('training/train_generator_loss', Gen_losses.avg, epoch)
    writer.add_scalar('training/train_discriminator_loss', Dis_losses.avg, epoch)
    writer.add_scalar('training/train_fake_loss', Fake_losses.avg, epoch)
    writer.add_scalar('training/train_real_loss', Real_losses.avg, epoch)


    # If the log weights argument is specified also add parameter and gradient histograms to TensorBoard.
    if args.log_weights:
        # Histograms and distributions of network parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins="auto")
            # second check required for buffers that appear in the parameters dict but don't receive gradients
            if value.requires_grad and value.grad is not None:
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins="auto")

    print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))