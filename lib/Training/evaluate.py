import torch


def get_latent_embedding(model, data_loader, num_classes, device):
    """
    Computes the latent embedding, i.e. z for each element of a dataset. The corresponding z values are directly
    organized by classes, such that they can readily be used for visualization purposes later.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Numbe of classes.
        device (str): Device to compute on.

    Returns:
        torch.Tensor: Stacked tensor of per class z vectors.
    """

    # switch to evaluation mode
    model.eval()

    zs = []
    for i in range(num_classes):
        zs.append([])

    # calculate probabilistic encoder for each dataset element. Store the corresponding mus and sigmas.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)

            encoded_mu, encoded_std = model.module.encode(inputs)
            z = model.module.reparameterize(encoded_mu, encoded_std)

            for i in range(inputs.size(0)):
                zs[classes[i]].append(z[i].data)

    # stack list of tensors into tensors
    for i in range(len(zs)):
        if len(zs[i]) > 0:
            zs[i] = torch.stack(zs[i], dim=0)

    return zs


def eval_dataset(model, data_loader, num_classes, device, samples=1):
    """
    Evaluates an entire dataset with the unified model and stores z values, latent mus and sigmas and output
    predictions according to whether the classification is correct or not.
    The values for correct predictions can later be used for plotting or fitting of Weibull models.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        samples (int): Number of variational samples.

    Returns:
        dict: Dictionary of results and latent values, separated by whether the classification was correct or not.
    """

    # switch to evaluation mode
    model.eval()

    correctly_identified = 0
    tot_samples = 0

    out_mus_correct = []
    out_sigmas_correct = []
    out_mus_false = []
    out_sigmas_false = []
    encoded_mus_correct = []
    encoded_mus_false = []
    encoded_sigmas_correct = []
    encoded_sigmas_false = []
    zs_correct = []
    zs_false = []

    for i in range(num_classes):
        out_mus_correct.append([])
        out_mus_false.append([])
        out_sigmas_correct.append([])
        out_sigmas_false.append([])
        encoded_mus_correct.append([])
        encoded_mus_false.append([])
        encoded_sigmas_correct.append([])
        encoded_sigmas_false.append([])
        zs_false.append([])
        zs_correct.append([])

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)
            encoded_mu, encoded_std = model.module.encode(inputs)

            out_samples = torch.zeros(samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(samples, encoded_mu.size(0), encoded_mu.size(1)).to(device)

            # sampling z and classifying
            for i in range(samples):
                z = model.module.reparameterize(encoded_mu, encoded_std)
                z_samples[i] = z

                cl = model.module.classifier(z)
                out = torch.nn.functional.softmax(cl, dim=1)
                out_samples[i] = out

            # calculate the mean and std. Only removes a dummy dimension if number of variational samples is set to one.
            out_mean = out_samples.mean(dim=0)
            out_std = out_samples.std(dim=0)
            zs_mean = z_samples.mean(dim=0)

            # for each input and respective prediction store independently depending on whether classification was
            # correct. The list of correct classifications is later used for fitting of Weibull models if the
            # data_loader is loading the training set.
            for i in range(inputs.size(0)):
                tot_samples += 1
                idx = torch.argmax(out_mean[i]).item()
                if classes[i].item() != idx:
                    out_mus_false[idx].append(out_mean[i][idx].item())
                    out_sigmas_false[idx].append(out_std[i][idx].item())
                    encoded_mus_false[idx].append(encoded_mu[i].data)
                    encoded_sigmas_false[idx].append(encoded_std[i].data)
                    zs_false[idx].append(zs_mean[i].data)
                else:
                    correctly_identified += 1
                    out_mus_correct[idx].append(out_mean[i][idx].item())
                    out_sigmas_correct[idx].append(out_std[i][idx].item())
                    encoded_mus_correct[idx].append(encoded_mu[i].data)
                    encoded_sigmas_correct[idx].append(encoded_std[i].data)
                    zs_correct[idx].append(zs_mean[i].data)

    acc = correctly_identified / float(tot_samples)

    # stack list of tensors into tensors
    for i in range(len(encoded_mus_correct)):
        if len(encoded_mus_correct[i]) > 0:
            encoded_mus_correct[i] = torch.stack(encoded_mus_correct[i], dim=0)
            encoded_sigmas_correct[i] = torch.stack(encoded_sigmas_correct[i], dim=0)
            zs_correct[i] = torch.stack(zs_correct[i], dim=0)
        if len(encoded_mus_false[i]) > 0:
            encoded_mus_false[i] = torch.stack(encoded_mus_false[i], dim=0)
            encoded_sigmas_false[i] = torch.stack(encoded_sigmas_false[i], dim=0)
            zs_false[i] = torch.stack(zs_false[i], dim=0)

    # Return a dictionary containing all the stored values
    return {"accuracy": acc, "encoded_mus_correct": encoded_mus_correct, "encoded_mus_false": encoded_mus_false,
            "encoded_sigmas_correct": encoded_sigmas_correct, "encoded_sigmas_false": encoded_sigmas_false,
            "zs_correct": zs_correct, "zs_false": zs_false,
            "out_mus_correct": out_mus_correct, "out_sigmas_correct": out_sigmas_correct,
            "out_mus_false": out_mus_false, "out_sigmas_false": out_sigmas_false}


def eval_openset_dataset(model, data_loader, num_classes, device, samples=1):
    """
    Evaluates an entire dataset with the unified model and stores z values, latent mus and sigmas and output
    predictions such that they can later be used for statistical outlier evaluation with the fitted Weibull models.
    This is merely for convenience to keep the rest of the code API the same. Note that the Weibull model's prediction
    of whether a sample from an unknown dataset is a statistical outlier or not can be done on an instance level.
    Similar to the eval_dataset function but without splitting of correct vs. false predictions as the dataset
    is unknown in the open-set scenario.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        samples (int): Number of variational samples.

    Returns:
        dict: Dictionary of results and latent values.
    """

    # switch to evaluation mode
    model.eval()

    out_mus = []
    out_sigmas = []
    encoded_mus = []
    encoded_sigmas = []
    zs = []

    for i in range(num_classes):
        out_mus.append([])
        out_sigmas.append([])
        encoded_mus.append([])
        encoded_sigmas.append([])
        zs.append([])

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)
            encoded_mu, encoded_std = model.module.encode(inputs)

            out_samples = torch.zeros(samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(samples, encoded_mu.size(0), encoded_mu.size(1)).to(device)

            # sampling z and classifying
            for i in range(samples):
                z = model.module.reparameterize(encoded_mu, encoded_std)
                z_samples[i] = z

                cl = model.module.classifier(z)
                out = torch.nn.functional.softmax(cl, dim=1)
                out_samples[i] = out

            # calculate the mean and std. This just removes a dummy dimension if variational samples are set to one.
            out_mean = out_samples.mean(0)
            out_std = out_samples.std(0)
            zs_mean = z_samples.mean(dim=0)

            # In contrast to the eval_dataset function, there is no split into correct or false values as the dataset
            # is unknown.
            for i in range(inputs.size(0)):
                idx = torch.argmax(out_mean[i]).item()
                out_mus[idx].append(out_mean[i][idx].item())
                out_sigmas[idx].append(out_std[i][idx].item())
                encoded_mus[idx].append(encoded_mu[i].data)
                encoded_sigmas[idx].append(encoded_std[i].data)
                zs[idx].append(zs_mean[i].data)

    # stack latent activations into a tensor
    for i in range(len(encoded_mus)):
        if len(encoded_mus[i]) > 0:
            encoded_mus[i] = torch.stack(encoded_mus[i], dim=0)
            encoded_sigmas[i] = torch.stack(encoded_sigmas[i], dim=0)
            zs[i] = torch.stack(zs[i], dim=0)

    # Return a dictionary of stored values.
    return {"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
            "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs}


def sample_per_class_zs(model, num_classes, num, device, use_new_z_bound, z_mean_bound):
    """
    Convenience method to draw samples from the prior and directly attach a label to them from the classifier.
    In generative replay with outlier rejection, the samples' outlier probabilities can then be directly computed and
    samples rejected or accepted before calculating the probabilistic decoder.

    We have added an optional flag and optional specification of Gaussian prior standard deviation, even though it is
    not described and has not been used in the original paper. The general idea is that if the optimization doesn't
    succeed and the approximate posterior is far away from a unit Gaussian, during generative replay we can gauge
    whether specific class regions of high density are sampled or not (as fitted by the Weibull models). If areas
    of large class density lie outside the range of a Unit Gaussian, we can adaptively change the prior because we have
    additional knowledge from the Weibull models on the expected range.

    Parameters:
        model (torch.nn.module): Trained model.
        num_classes (int): Number of classes.
        num (int): Number of samples to draw.
        device (str): Device to compute on.
        use_new_z_bound (bool): Flag indicating whether a modifed prior with larger std should be used.
        z_mean_bound (float): New standard deviation for the Gaussian prior if use_new_z_bound is True.
    """
    
    z_samples_per_class = []
    for i in range(num_classes):
        z_samples_per_class.append([])

    # sample from the prior or modified prior.
    if use_new_z_bound:
        z_samples = torch.randn(num, model.module.latent_dim).to(device) * z_mean_bound
    else:
        z_samples = torch.randn(num, model.module.latent_dim).to(device)

    # classify the samples
    cl = model.module.classifier(z_samples)
    out = torch.nn.functional.softmax(cl, dim=1)

    # Store the z samples and labels.
    for i in range(out.size(0)):
        idx = torch.argmax(out[i]).item()
        z_samples_per_class[idx].append(z_samples[i].data)

    for i in range(len(z_samples_per_class)):
        if len(z_samples_per_class[i]) > 0:
            z_samples_per_class[i] = torch.stack(z_samples_per_class[i], dim=0)

    return {"z_samples": z_samples_per_class}
