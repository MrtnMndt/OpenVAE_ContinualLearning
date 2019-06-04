import math
import os
import torchvision
import torch.utils.data
from tqdm import tqdm
from tqdm import trange
import lib.OpenSet.meta_recognition as mr
from lib.Training.evaluate import sample_per_class_zs
import lib.Datasets.datasets as all_datasets
from lib.Training.evaluate import eval_dataset


def get_incremental_dataset(parent_class, args):
    """
    Wrapper that returns either the incremental dataset or cross-dataset class. For the incremental dataset scenario
    the class will inherit from the original dataset and split the dataset as specified in the command line arguments.

    We note that the use of the term task is slightly ambiguous in the two classes but have decided to use a single
    command line parameter nevertheless. In the class incremental scenario number of tasks refers to number of classes,
    in the cross-dataset scenario number of tasks refers to number of datasets. As our model is a unified model with a
    growing classifier it technically always only has one task, namely to classify, reconstruct and replay everything
    it has seen so far. In our context the below task variables arethus used for convenience to indicate when something
    is added and how much has been seen already. We hope that this doesn't cause too much confusion.

    Parameters:
        parent_class: Dataset class to inherit from for the class incremental scenario.
        args (dict): Command line arguments.
    """

    class IncrementalDataset(parent_class):
        """
        Incremental dataset class. Inherits from a dataset parent class. Defines functions to split classes into
        separate sets, incrementing the current set and replacing previous sets with generative replay examples.

        Parameters:
        is_gpu (bool): True if CUDA is enabled. Sets value of pin_memory in DataLoader.
        task_order (list): List defining class order (sequence of integers).
        args (dict): Dictionary of (command line) arguments. Needs to contain num_base_tasks (int),
            num_increment_tasks (int), batch_size (int), workers(int), var_samples (int) and distance_function (str).

        Attributes:
            task_order (list): Sequence of integers defining incremental class ordering.
            seen_tasks (list): List of already seen tasks at any given increment.
            num_base_tasks (int): Number of initial classes.
            num_increment_tasks (int): Amount of classes that get added with each increment.
            device (str): Device to compute on
            vis_size (int): Visualization size used in generation of dataset snapshots.
            trainsets (torch.utils.data.TensorDataset): Training set wrapper.
            trainset (torch.utils.data.TensorDataset): Training increment set wrapper.
            valsets (torch.utils.data.TensorDataset): Validation set wrapper
            valset (torch.utils.data.TensorDataset): Validation increment set wrapper.
            class_to_idx (dict): Defines mapping from class names to integers.
            train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
            val_loader (torch.utils.data.DataLoader): Validation set loader.
        """

        def __init__(self, is_gpu, device, task_order, args):
            super().__init__(is_gpu, args)

            self.task_order = task_order
            self.seen_tasks = []
            self.num_base_tasks = args.num_base_tasks + 1
            self.num_increment_tasks = args.num_increment_tasks
            self.device = device
            self.args = args

            self.vis_size = 144

            self.trainsets, self.valsets = {}, {}

            self.class_to_idx = {}

            # Split the parent dataset class into into datasets per class
            self.__get_incremental_datasets()

            # Get the corresponding class datasets for the initial datasets as specified by number and order
            self.trainset, self.valset = self.__get_initial_dataset()
            # Get the respective initial class data loaders
            self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        def __get_incremental_datasets(self):
            """
            Splits the existing parent dataset into separate datasets per class. As our model's use a single-head
            growing classifier, also relabels the targets according to task sequence so the first encountered class
            is always 0, the second 1, etc. even if the former label was something else like class 5 and 7.
            """
            datasets = [self.trainset, self.valset]
            # once for train and once for valset
            for j in range(2):
                tensors_list, targets_list = [], []
                for i in range(self.num_classes):
                    tensors_list.append([])
                    targets_list.append([])
                # loop through the entire dataset
                for i, (inp, target) in enumerate(datasets[j]):
                    # because data loaders (especially from torchvision) can provide targets in different formats,
                    # we include a potential conversion step from e.g. one-hot to integer as this is what our loss
                    # functions will use.
                    if isinstance(target, int):
                        pass
                    elif isinstance(target, torch.Tensor):
                        if target.dim() > 0:
                            target = int(torch.argmax(target))
                        else:
                            target = int(target)
                    else:
                        raise ValueError

                    # Relabel the dataset according to task sequence. I.e. if the first two classes are 5 and 7,
                    # they are now relabeled 5->0 and 7->1. We do this so we can be agnostic to task ordering and
                    # because the growing classifier at any time only has the corresponding amount of units
                    # (e.g. in the example case of 5,7 it has only 2 units at the beginning)
                    relabeled_target = self.task_order.index(target)

                    tensors_list[target].append(inp)
                    targets_list[target].append(relabeled_target)

                # For each class, compose a separate tensor dataset that we can join later
                for i in range(self.num_classes):
                    tensors_list[i] = torch.stack(tensors_list[i], dim=0)
                    targets_list[i] = torch.LongTensor(targets_list[i])

                    if j == 0:
                        self.trainsets[i] = torch.utils.data.TensorDataset(tensors_list[i], targets_list[i])
                    else:
                        self.valsets[i] = torch.utils.data.TensorDataset(tensors_list[i], targets_list[i])

        def __get_initial_dataset(self):
            """
            Fills the initial trainset and valset for the number of base tasks/classes in the order specified by the
            task order attribute.

            Returns:
                torch.utils.data.TensorDataset: trainset, valset
            """

            # Pop the initial num_base_tasks many class indices from the task order and fill seen_tasks.
            for i in range(self.num_base_tasks):
                self.class_to_idx[str(self.task_order[0])] = i
                self.seen_tasks.append(self.task_order.pop(0))

            # Join/Concatenate the separate class datasets for the initial base tasks according to the just filled
            # seen_tasks list.
            trainset = torch.utils.data.ConcatDataset([self.trainsets[j] for j in self.seen_tasks])
            valset = torch.utils.data.ConcatDataset([self.valsets[j] for j in self.seen_tasks])

            # Also pop the trainsets and valsets that were just concatenated so the next increment cannot use them
            # again. Because the order is random and can be e.g. increasing, we pop back to front
            sorted_tasks = sorted(self.seen_tasks, reverse=True)
            for i in sorted_tasks:
                self.trainsets.pop(i)
                self.valsets.pop(i)

            return trainset, valset

        def get_dataset_loader(self, batch_size, workers, is_gpu):
            """
            Defines the dataset loader for wrapped dataset

            Parameters:
                batch_size (int): Defines the batch size in data loader
                workers (int): Number of parallel threads to be used by data loader
                is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

            Returns:
                 torch.utils.data.DataLoader: train_loader, val_loader
            """

            train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,
                                                       num_workers=workers, pin_memory=is_gpu, sampler=None)

            val_loader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, shuffle=True,
                                                     num_workers=workers, pin_memory=is_gpu)

            return train_loader, val_loader

        def increment_tasks(self, model, batch_size, workers, writer, save_path, is_gpu,
                            upper_bound_baseline=False, generative_replay=False, openset_generative_replay=False,
                            openset_threshold=0.05, openset_tailsize=0.05, autoregression=False):
            """
            Main function to increment tasks/classes. Has multiple options to specify whether the new dataset should
            provide an upper bound for a continual learning experiment by concatenation of new real data with
            existing real data, using generative replay to rehearse previous tasks or using the OCDVAE's generative
            replay with statistical outlier rejection to rehearse previous tasks. If nothing is specified the
            incremental lower bound of just training on the new task increment is considered. Validation sets are always
            composed of real data as well as the newly added task increment that is yet unseen. The latter is added by
            popping indices from the task order queue and adding corresponding individual datasets.

            Overwrites trainset and valset and updates the classes' train and val loaders.

            Parameters:
                model (torch.nn.module): unified model, needed if any form of generative replay is active.
                batch_size (int): batch_size for generative replay. Only defines computation speed of replay.
                workers (int): workers for parallel cpu thread for the newly composed data loaders.
                writer (tensorboard.SummaryWriter): TensorBoard writer instance.
                save_path (str): Path used for saving snapshots.
                is_gpu (bool): Flag indicating whether GPU is used. Needed for the pin memory of the data loaders.
                upper_bound_baseline (bool): If on, real data is kept and concatenated with new task's real data.
                generative replay (bool): If True, generative replay is used to rehearse previously seen tasks.
                openset_generative_replay (bool): If True, generative replay with statistical outlier rejection is
                    used, corresponding to algorithm 3 of the paper and the OCDVAE model.
                openset_threshold (float): statistical outlier rejection prior that is used to reject z samples from
                    regions of low density.
                openset_tailsize (int): Tailsize used in the fit of the Weibul models. Should be a percentage
                    (range 0-1) specifying amount of dataset expected to be considered as atypical. Typically this
                    is something low like 5% or even less.
                autoregression (bool): If True, generative replay is conducted with the autoregressive model.
            """

            # pop the next num_increment_tasks many task indices from the task order list and add them to seen tasks
            new_tasks = []
            for i in range(self.num_increment_tasks):
                idx = self.task_order.pop(0)
                new_tasks.append(idx)
                # also construct the new class_to_idx table for the confusion matrix (so we later see e.g. class 7 at
                # the first place etc.
                self.class_to_idx[str(idx)] = len(self.seen_tasks)
                self.seen_tasks.append(idx)

            # again, sort the new tasks so we can pop back to front.
            sorted_new_tasks = sorted(new_tasks, reverse=True)

            if upper_bound_baseline:
                # concatenate new task with all previously seen tasks
                new_trainsets = [self.trainsets.pop(j) for j in sorted_new_tasks]
                new_trainsets.append(self.trainset)
                self.trainset = torch.utils.data.ConcatDataset(new_trainsets)
            elif generative_replay or openset_generative_replay:
                # use generative model to replay old tasks and concatenate with new task's real data
                new_trainsets = [self.trainsets.pop(j) for j in sorted_new_tasks]

                genset = self.generate_seen_tasks(model, batch_size, len(self.trainset), writer, save_path,
                                                  openset=openset_generative_replay,
                                                  openset_threshold=openset_threshold,
                                                  openset_tailsize=openset_tailsize,
                                                  autoregression=autoregression)
                new_trainsets.append(genset)
                self.trainset = torch.utils.data.ConcatDataset(new_trainsets)
            else:
                # only see new task's real data (incremental lower bound)
                if self.num_increment_tasks == 1:
                    self.trainset = self.trainsets.pop(new_tasks[0])
                else:
                    self.trainset = torch.utils.data.ConcatDataset([self.trainsets.pop(j) for j in sorted_new_tasks])

            # get the new validation sets and concatenate them to existing validation data.
            # note that validation is always conducted on real data, while training can be done on generated samples.
            new_valsets = [self.valsets.pop(j) for j in sorted_new_tasks]
            new_valsets.append(self.valset)
            self.valset = torch.utils.data.ConcatDataset(new_valsets)

            # update/overwrite the train and val loaders
            self.train_loader, self.val_loader = self.get_dataset_loader(batch_size, workers, is_gpu)

        def generate_seen_tasks(self, model, batch_size, seen_dataset_size, writer, save_path,
                                openset=False, openset_threshold=0.05, openset_tailsize=0.05, autoregression=False):
            """
            The function implementing the actual generative replay and openset generative replay with statistical
            outlier rejection.

            Parameters:
                model (torch.nn.module): Unified model, needed if any form of generative replay is active.
                batch_size (int): Batch_size for generative replay. Only defines computation speed of replay.
                seen_dataset_size (int): Number of data points to generate. As the name suggests we have set this
                    to the exact number of previously seen real data points. In principle this can be a hyper-parameter.
                writer (tensorboard.SummaryWriter): TensorBoard writer instance.
                save_path (str): Path used for saving snapshots.
                openset_generative_replay (bool): If True, generative replay with statistical outlier rejection is
                    used, corresponding to algorithm 3 of the paper and the OCDVAE model.
                openset_threshold (float): statistical outlier rejection prior that is used to reject z samples from
                    regions of low density.
                openset_tailsize (int): Tailsize used in the fit of the Weibul models. Should be a percentage
                    (range 0-1) specifying amount of dataset expected to be considered as atypical. Typically this
                    is something low like 5% or even less.
                autoregression (bool): If True, generative replay is conducted with the autoregressive model.

            Returns:
                torch.utils.data.TensorDataset: generated trainset
            """

            data = []
            zs = []
            targets = []

            # flag to default to regular generative replay if openset fit is not successful
            openset_success = True
            # if the openset flag is active, we proceed with generative replay with statistical outlier rejection,
            # i.e. the OCDVAE, if not we continue with conventional generative replay where every sample is a simple
            # draw from the Unit Gaussian prior, i.e. a CDVAE.
            if openset:
                # Start with fitting the Weibull functions based on the available train data and the classes seen
                # so far. Note that if generative replay has been called before after the first task increment,
                # this means that previous train data consists of already generated data.
                # Evaluate the training dataset to find the correctly classified examples.
                dataset_train_dict = eval_dataset(model, self.train_loader,
                                                  len(self.seen_tasks) - self.num_increment_tasks, self.device,
                                                  samples=self.args.var_samples)
                # Find the per class mean of z, i.e. the class specific regions of highest density of the approximate
                # posterior.
                z_means = mr.get_means(dataset_train_dict["zs_correct"])

                # get minimum and maximum values in case there is class clusters lying outside of the standard Gaussian
                # range. This can be the case if the approximate posterior deviates a lot from the prior.
                # If sampling from a standard Normal distribution fails, these values will be used in a second attempt
                # at sampling. Note that we have not used this in the paper (as it never occurred) but add this as it
                # could potentially aid users in the future.
                use_new_z_bound = False
                z_mean_bound = -100000
                for c in range(len(z_means)):
                    if isinstance(z_means[c], torch.Tensor):
                        tmp_bound = torch.max(torch.abs(z_means[c])).cpu().item()
                        if tmp_bound > z_mean_bound:
                            z_mean_bound = tmp_bound

                # Calculate the correctly classified data point's distance in latent space to the per class mean zs.
                train_distances_to_mu = mr.calc_distances_to_means(z_means, dataset_train_dict["zs_correct"],
                                                                   self.args.distance_function)
                # determine tailsize according to set percentage and dataset size
                tailsize = int((seen_dataset_size * openset_tailsize) /
                               (len(self.seen_tasks) - self.num_increment_tasks))
                print("Fitting Weibull models with tailsize: " + str(tailsize))
                # set the tailsize per class (assuming a balanced dataset)
                tailsizes = [tailsize] * (len(self.seen_tasks) - self.num_increment_tasks)
                # fit the weibull models
                weibull_models, valid_weibull = mr.fit_weibull_models(train_distances_to_mu, tailsizes)

                if not valid_weibull:
                    print("Open set fit was not successful")
                    openset_success = False
                else:
                    print("Using generative model to replay old data with openset detection")
                    # set class counters to count amount of generations
                    class_counters = [0] * (len(self.seen_tasks) - self.num_increment_tasks)
                    # seen tasks minus incremented tasks
                    # because seen tasks have already been incremented in the function calling this method,
                    # generating equal number of samples for each of the previously seen tasks
                    samples_per_class = int(math.ceil(seen_dataset_size /
                                                      (len(self.seen_tasks) - self.num_increment_tasks)))
                    openset_attempts = 0

                    # progress bar
                    pbar = tqdm(total=seen_dataset_size)
                    # as long as the desired generated dataset size is not reached, continue
                    while sum(class_counters) < seen_dataset_size:
                        # sample zs and classify them. Sort them according to classes
                        z_dict = sample_per_class_zs(model, len(self.seen_tasks) - self.num_increment_tasks,
                                                     batch_size, self.device, use_new_z_bound, z_mean_bound)
                        # Calculate the distance of each z to the per class mean z.
                        z_samples_distances_to_mean = mr.calc_distances_to_means(z_means, z_dict["z_samples"],
                                                                                 self.args.distance_function)
                        # Evaluate the the statistical outlier probability using the respective Weibull model
                        z_samples_outlier_probs = mr.calc_outlier_probs(weibull_models, z_samples_distances_to_mean)

                        # For each class reject or accept the samples based on outlier probability and chosen prior.
                        for i in range(len(self.seen_tasks) - self.num_increment_tasks):
                            # only add images if per class sample amount hasn't been surpassed yet in order
                            # to balance the dataset
                            for j in range(len(z_samples_outlier_probs[i])):
                                # check the openset threshold for each example and only add if generation is not
                                # classified as openset outlier (i.e. somewhere in "unseen" latent space)
                                if class_counters[i] < samples_per_class:
                                    if z_samples_outlier_probs[i][j] < openset_threshold:
                                        zs.append(z_dict["z_samples"][i][j])
                                        targets.append(i)
                                        class_counters[i] += 1
                                        pbar.update(1)
                                else:
                                    break
                        # increment the number of open set attempts
                        openset_attempts += 1

                        # time out if none of the samples pass the above test. This can happen if either the
                        # approximate posterior has not been optimized properly and is very far away from the prior
                        # or if the rejection prior has been set to something extremely small (like 0.01%).
                        if openset_attempts == 2000 and any([val == 0 for val in class_counters]):
                            # reset samples
                            data = []
                            zs = []
                            targets = []

                            # if sampling from standard Gaussian failed the first time, try with different prior std
                            if use_new_z_bound:
                                print("\n Open set generative replay timeout")
                                openset_success = False
                                break
                            else:
                                print("\n Open set generative replay from standard Gaussian failed. Trying sampling "
                                      "with modified variance bound")
                                use_new_z_bound = True
                                openset_attempts = 0

                    pbar.close()

                    # once all the samples from the prior have been accepted to fill the entire dataset size we
                    # proceed with generating the actual data points using the probabilistic decoder.
                    if openset_success:
                        print("Openset sampling successful. Generating dataset")
                        zs = torch.stack(zs, dim=0)
                        targets = torch.LongTensor(targets)

                        # actually generate images from valid zs
                        for i in trange(0, len(zs), batch_size):
                            gen = model.module.decode(zs[i:i + batch_size])
                            gen = torch.sigmoid(gen)
                            if autoregression:
                                gen = model.module.pixelcnn.generate(gen)
                            data.append(gen.data.cpu())
                        data = torch.cat(data, dim=0)

                        # get a subset for visualization and distribute it evenly along classes
                        _, sd_idx = torch.sort(targets)
                        subset_idx = sd_idx[torch.floor(torch.arange(0, data.size(0),
                                                                     data.size(0) / self.vis_size)).long()]
                        viz_subset = data[subset_idx]

                        imgs = torchvision.utils.make_grid(viz_subset, nrow=int(math.sqrt(self.vis_size)),
                                                           padding=5)
                        torchvision.utils.save_image(viz_subset, os.path.join(save_path, 'samples_seen_tasks_' +
                                                                              str(len(self.seen_tasks) -
                                                                                  self.num_increment_tasks) + '.png'),
                                                     nrow=int(math.sqrt(self.vis_size)), padding=5)
                        writer.add_image('openset_generation_snapshot', imgs, len(self.seen_tasks) -
                                         self.num_increment_tasks)

                        # return the new trainset.
                        trainset = torch.utils.data.TensorDataset(data, targets)
                        return trainset

            # If openset generative replay with outlier rejection has failed (e.g. rejection prior set to something
            # very close to zero, model not having trained at all or approximate posterior deviating extremely from the
            # prior) or isn't desired, conventional generative replay is conducted.
            if not openset or not openset_success:
                print("Using generative model to replay old data")
                for i in trange(int(seen_dataset_size / batch_size)):
                    # sample from the prior
                    z_samples = torch.randn(batch_size, model.module.latent_dim).to(self.device)

                    # calculate probabilistic decoder, generate data points
                    gen = model.module.decode(z_samples)
                    gen = torch.sigmoid(gen)
                    if autoregression:
                        gen = model.module.pixelcnn.generate(gen)

                    # classify the samples from the prior and set the label correspondingly.
                    cl = model.module.classifier(z_samples)
                    cl = torch.nn.functional.softmax(cl, dim=1)
                    label = torch.argmax(cl, dim=1)
                    data.append(gen.data.cpu())
                    targets.append(label.data.cpu())

                # need to detach from graph as otherwise the "require grad" will crash auto differentiation
                data = torch.cat(data, dim=0)
                targets = torch.cat(targets, dim=0)

                # get a subset for visualization
                _, sd_idx = torch.sort(targets)
                subset_idx = sd_idx[torch.floor(torch.arange(0, data.size(0), data.size(0) / self.vis_size)).long()]
                viz_subset = data[subset_idx]

                imgs = torchvision.utils.make_grid(viz_subset, nrow=int(math.sqrt(self.vis_size)), padding=5)
                torchvision.utils.save_image(viz_subset, os.path.join(save_path, 'samples_seen_tasks_' +
                                                                      str(len(self.seen_tasks) -
                                                                          self.num_increment_tasks) + '.png'),
                                             nrow=int(math.sqrt(self.vis_size)), padding=5)
                writer.add_image('dataset_generation_snapshot', imgs, len(self.seen_tasks) - self.num_increment_tasks)

            # return generated trainset
            trainset = torch.utils.data.TensorDataset(data, targets)
            return trainset

    class CrossDataset:
        """
        Cross-dataset class. Defines functions to split and join datasets, incrementing the current set
        and replacing previous sets with generative replay examples.

        The diffences to above incremental classes are mainly in how task numbers are set and datasets are
        acquired. The logic in increments or generative replay is analogous. This class could be merged with the
        class incremental dataset class, but we have chosen to keep them separate to avoid too many conditions in the
        code, even if this means that the class contains a lot of copy paste code.

        Parameters:
        is_gpu (bool): True if CUDA is enabled. Sets value of pin_memory in DataLoader.
        task_order (str): String, defining a comma separate sequence of dataset names.
        args (dict): Dictionary of (command line) arguments. Needs to contain num_base_tasks (int),
            num_increment_tasks (int), batch_size (int), workers(int), var_samples (int) and distance_function (str).

        Attributes:
            task_order (str): String, defining a comma separate sequence of dataset names.
            seen_tasks (str): String, defining a sequence of already seen dataset names.
            num_base_tasks (int): Number of initial datasets.
            num_increment_tasks (int): Amount of classes that get added with each increment.
            device (str): Device to compute on
            vis_size (int): Visualization size used in generation of dataset snapshots.
            trainsets (torch.utils.data.TensorDataset): Training set wrapper.
            trainset (torch.utils.data.TensorDataset): Training increment set wrapper.
            valsets (torch.utils.data.TensorDataset): Validation set wrapper
            valset (torch.utils.data.TensorDataset): Validation increment set wrapper.
            class_to_idx (dict): Defines mapping from class names to integers.
            train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
            val_loader (torch.utils.data.DataLoader): Validation set loader.
        """

        def __init__(self, is_gpu, device, task_order, args):
            self.task_order = task_order
            self.num_classes_per_task = []
            self.seen_tasks = []
            self.num_base_tasks = args.num_base_tasks
            self.num_increment_tasks = args.num_increment_tasks
            self.device = device
            self.args = args

            self.vis_size = 144

            self.trainsets, self.valsets = {}, {}
            self.num_images_per_dataset = [0]

            self.task_to_idx = {}

            self.__get_incremental_datasets()

            self.trainset, self.valset = self.__get_initial_dataset()
            self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        def __get_incremental_datasets(self):
            """
            Gets the dataset for each dataset specified in the task order sequence. Relabels the targets according to
            task sequence so the first encountered dataset has classes starting from 0 up to the number of classes in
            that dataset and the next dataset starts with its first class incremented by that number.
            """
            trainsets = []
            valsets = []
            # for each dataset in task order, get the corresponding dataset class, create an instance and
            # get the dataset. Set the number of classes for each task/dataset so we can infer it later.
            for i, dataset_name in enumerate(self.task_order):
                temp_dataset_init_method = getattr(all_datasets, dataset_name)
                temp_dataset = temp_dataset_init_method(torch.cuda.is_available(), self.args)
                trainsets.append(temp_dataset.trainset)
                valsets.append(temp_dataset.valset)
                self.num_classes_per_task.append(temp_dataset.num_classes)
                del temp_dataset
                # changing task_order from a list of strings to a list of numbers to keep in sync
                # with the class-incremental code
                self.task_order[i] = i

            # once for train and once for valset, similar to the class incremental class
            datasets = [trainsets, valsets]
            for j in range(2):
                tensors_list, targets_list = [], []
                for i in range(len(self.task_order)):
                    tensors_list.append([])
                    targets_list.append([])
                    # loop through the entire dataset
                    for k, (inp, target) in enumerate(datasets[j][i]):
                        # because data loaders (especially from torchvision) can provide targets in different formats,
                        # we include a potential conversion step from e.g. one-hot to integer as this is what our loss
                        # functions will use.
                        if isinstance(target, int):
                            pass
                        elif isinstance(target, torch.Tensor):
                            if target.dim() > 0:
                                target = int(torch.argmax(target))
                            else:
                                target = int(target)
                        else:
                            raise ValueError

                        # Relabel the dataset according to task sequence, i.e. the labels for the second dataset will
                        # start where the first one ended..
                        relabeled_target = sum(self.num_classes_per_task[:i]) + target

                        tensors_list[i].append(inp)
                        targets_list[i].append(relabeled_target)

                for i in range(len(self.task_order)):
                    tensors_list[i] = torch.stack(tensors_list[i], dim=0)
                    targets_list[i] = torch.LongTensor(targets_list[i])

                    # For each dataset, compose a separate tensor dataset that we can join later
                    if j == 0:
                        self.trainsets[i] = torch.utils.data.TensorDataset(tensors_list[i], targets_list[i])
                    else:
                        self.valsets[i] = torch.utils.data.TensorDataset(tensors_list[i], targets_list[i])

        def __get_initial_dataset(self):
            """
            Fills the initial trainset and valset for the number of base tasks/classes in the order specified by the
            task order attribute.

            Returns:
                torch.utils.data.TensorDataset: trainset, valset
            """
            # pop the initial number of datasets, add them to seen task and set the class indices.
            for i in range(self.num_base_tasks):
                self.seen_tasks.append(self.task_order.pop(0))
            for i in range(sum(self.num_classes_per_task[: self.num_base_tasks])):
                self.task_to_idx[i] = i

            # Concatenate the datasets (if more than one initial dataset is requested)
            trainset = torch.utils.data.ConcatDataset([self.trainsets[j] for j in self.seen_tasks])
            valset = torch.utils.data.ConcatDataset([self.valsets[j] for j in self.seen_tasks])

            # sort to pop back to front
            sorted_tasks = sorted(self.seen_tasks, reverse=True)
            for i in sorted_tasks:
                self.trainsets.pop(i)
                self.valsets.pop(i)

            return trainset, valset

        def get_dataset_loader(self, batch_size, workers, is_gpu):
            """
            Defines the dataset loader for wrapped dataset

            Parameters:
                batch_size (int): Defines the batch size in data loader
                workers (int): Number of parallel threads to be used by data loader
                is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

            Returns:
                 torch.utils.data.TensorDataset: trainset, valset
            """

            train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,
                                                       num_workers=workers, pin_memory=is_gpu, sampler=None)

            val_loader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, shuffle=True,
                                                     num_workers=workers, pin_memory=is_gpu)

            return train_loader, val_loader

        def increment_tasks(self, model, batch_size, workers, writer, save_path, is_gpu,
                            upper_bound_baseline=False, generative_replay=False, openset_generative_replay=False,
                            openset_threshold=0.2, openset_tailsize=0.05, autoregression=False):
            """
            Main function to increment tasks/datasets. Has multiple options to specify whether the new dataset should
            provide an upper bound for a continual learning experiment by concatenation of new real data with
            existing real data, using generative replay to rehearse previous tasks or using the OCDVAE's generative
            replay with statistical outlier rejection to rehearse previous tasks. If nothing is specified the
            incremental lower bound of just training on the new task increment is considered. Validation sets are always
            composed of real data as well as the newly added task increment that is yet unseen. The latter is added by
            popping indices from the task order queue and adding corresponding individual datasets.

            Overwrites trainset and valset and updates the classes' train and val loaders.

            Parameters:
                model (torch.nn.module): unified model, needed if any form of generative replay is active.
                batch_size (int): batch_size for generative replay. Only defines computation speed of replay.
                workers (int): workers for parallel cpu thread for the newly composed data loaders.
                writer (tensorboard.SummaryWriter): TensorBoard writer instance.
                save_path (str): Path used for saving snapshots.
                is_gpu (bool): Flag indicating whether GPU is used. Needed for the pin memory of the data loaders.
                upper_bound_baseline (bool): If on, real data is kept and concatenated with new task's real data.
                generative replay (bool): If True, generative replay is used to rehearse previously seen tasks.
                openset_generative_replay (bool): If True, generative replay with statistical outlier rejection is
                    used, corresponding to algorithm 3 of the paper and the OCDVAE model.
                openset_threshold (float): statistical outlier rejection prior that is used to reject z samples from
                    regions of low density.
                openset_tailsize (int): Tailsize used in the fit of the Weibul models. Should be a percentage
                    (range 0-1) specifying amount of dataset expected to be considered as atypical. Typically this
                    is something low like 5% or even less.
                autoregression (bool): If True, generative replay is conducted with the autoregressive model.
            """

            # store length of old/current dataset
            self.num_images_per_dataset.append(len(self.trainset))

            new_tasks = []
            # pop the next num_increment_tasks many task indices (datasets) from the task order list and
            # add them to seen tasks
            for i in range(self.num_increment_tasks):
                idx = self.task_order.pop(0)
                new_tasks.append(idx)
                self.seen_tasks.append(idx)
            # also construct the new task_to_idx table for the confusion matrix)
            for i in range(sum(self.num_classes_per_task[:len(self.seen_tasks) - args.num_increment_tasks]),
                           sum(self.num_classes_per_task[:len(self.seen_tasks)])):
                self.task_to_idx[i] = i

            # again, sort the new tasks so we can pop back to front.
            sorted_new_tasks = sorted(new_tasks, reverse=True)

            if upper_bound_baseline:
                # concatenate new task with all previously seen tasks
                new_trainsets = [self.trainsets.pop(j) for j in sorted_new_tasks]
                new_trainsets.append(self.trainset)
                self.trainset = torch.utils.data.ConcatDataset(new_trainsets)
            elif generative_replay or openset_generative_replay:
                # use generative model to replay old tasks and concatenate with new task's real data
                new_trainsets = [self.trainsets.pop(j) for j in sorted_new_tasks]

                genset = self.generate_seen_tasks(model, batch_size, len(self.trainset), writer, save_path,
                                                  openset=openset_generative_replay,
                                                  openset_threshold=openset_threshold,
                                                  openset_tailsize=openset_tailsize,
                                                  autoregression=autoregression)
                new_trainsets.append(genset)
                self.trainset = torch.utils.data.ConcatDataset(new_trainsets)
            else:
                # only see new task's real data (incremental lower bound)
                if self.num_increment_tasks == 1:
                    self.trainset = self.trainsets.pop(new_tasks[0])
                else:
                    self.trainset = torch.utils.data.ConcatDataset([self.trainsets.pop(j) for j in sorted_new_tasks])

            # get the new validation sets and concatenate them to existing validation data.
            # note that validation is always conducted on real data, while training can be done on generated samples.
            new_valsets = [self.valsets.pop(j) for j in sorted_new_tasks]
            new_valsets.append(self.valset)
            self.valset = torch.utils.data.ConcatDataset(new_valsets)

            # update/overwrite the train and val loaders
            self.train_loader, self.val_loader = self.get_dataset_loader(batch_size, workers, is_gpu)

        def generate_seen_tasks(self, model, batch_size, seen_dataset_size, writer, save_path,
                                openset=False, openset_threshold=0.2, openset_tailsize=0.05,
                                autoregression=False):
            """
            The function implementing the actual generative replay and openset generative replay with statistical
            outlier rejection. The only real difference with generative replay as specified in the class incremental
            dataset class is in the way dataset lengths, tailsizes, number of classes are set as we deal with entire
            datasets here.

            Parameters:
                model (torch.nn.module): Unified model, needed if any form of generative replay is active.
                batch_size (int): Batch_size for generative replay. Only defines computation speed of replay.
                seen_dataset_size (int): Number of data points to generate. As the name suggests we have set this
                    to the exact number of previously seen real data points. In principle this can be a hyper-parameter.
                writer (tensorboard.SummaryWriter): TensorBoard writer instance.
                save_path (str): Path used for saving snapshots.
                openset_generative_replay (bool): If True, generative replay with statistical outlier rejection is
                    used, corresponding to algorithm 3 of the paper and the OCDVAE model.
                openset_threshold (float): statistical outlier rejection prior that is used to reject z samples from
                    regions of low density.
                openset_tailsize (int): Tailsize used in the fit of the Weibul models. Should be a percentage
                    (range 0-1) specifying amount of dataset expected to be considered as atypical. Typically this
                    is something low like 5% or even less.
                autoregression (bool): If True, generative replay is conducted with the autoregressive model.

            Returns:
                torch.utils.data.TensorDataset: generated trainset
            """

            data = []
            zs = []
            targets = []

            # flag to default to regular generative replay if openset fit is not successful
            openset_success = True
            # if the openset flag is active, we proceed with generative replay with statistical outlier rejection,
            # i.e. the OCDVAE, if not we continue with conventional generative replay where every sample is a simple
            # draw from the Unit Gaussian prior, i.e. a CDVAE.
            if openset:
                # Start with fitting the Weibull functions based on the available train data and the classes seen
                # so far. Note that if generative replay has been called before after the first task increment,
                # this means that previous train data consists of already generated data.
                # Evaluate the training dataset to find the correctly classified examples.
                num_seen_classes = sum(self.num_classes_per_task[:len(self.seen_tasks) - self.args.num_increment_tasks])

                dataset_train_dict = eval_dataset(model, self.train_loader, num_seen_classes, self.device,
                                                  samples=self.args.var_samples)

                # Find the per class mean of z, i.e. the class specific regions of highest density of the approximate
                # posterior.
                z_means = mr.get_means(dataset_train_dict["zs_correct"])

                # get minimum and maximum values in case there is class clusters lying outside of the standard Gaussian
                # range. This can be the case if the approximate posterior deviates a lot from the prior.
                # If sampling from a standard Normal distribution fails, these values will be used in a second attempt
                # at sampling. Note that we have not used this in the paper (as it never occurred) but add this as it
                # could potentially aid users in the future.
                use_new_z_bound = False
                z_mean_bound = -100000
                for c in range(len(z_means)):
                    if isinstance(z_means[c], torch.Tensor):
                        tmp_bound = torch.max(torch.abs(z_means[c])).cpu().item()
                        if tmp_bound > z_mean_bound:
                            z_mean_bound = tmp_bound

                # Calculate the correctly classified data point's distance in latent space to the per class mean zs.
                train_distances_to_mu = mr.calc_distances_to_means(z_means, dataset_train_dict["zs_correct"],
                                                                   self.args.distance_function)
                # # determine tailsize according to set percentage and dataset size. As in the incremental
                # class scenario it is assumed that the number of samples per class per dataset is balanced.
                tailsizes = []
                for i in range(len(self.num_images_per_dataset) - 1):
                    tailsizes.append([int(((self.num_images_per_dataset[i+1] - self.num_images_per_dataset[i]) *
                                           openset_tailsize) / self.num_classes_per_task[i])] *
                                     self.num_classes_per_task[i])
                # extend the tailsize for each dataset to all of its classes (assuming that each dataset is balanced)
                tailsizes = [item for sublist in tailsizes for item in sublist]
                print("Fitting Weibull models with tailsizes: ", tailsizes)
                # fit the weibull models
                weibull_models, valid_weibull = mr.fit_weibull_models(train_distances_to_mu, tailsizes)

                if not valid_weibull:
                    print("Open set fit was not successful")
                    openset_success = False
                else:
                    print("Using generative model to replay old data with openset detection")
                    # set class counters to count amount of generations
                    class_counters = [0] * num_seen_classes

                    # calculate number of samples per class according to original dataset sizes
                    samples_per_class = []
                    for i in range(len(self.num_images_per_dataset) - 1):
                        samples_per_class.append([int(math.ceil((self.num_images_per_dataset[i+1] -
                                                                 self.num_images_per_dataset[i]) /
                                                                self.num_classes_per_task[i]))] *
                                                 self.num_classes_per_task[i])
                    samples_per_class = [item for sublist in samples_per_class for item in sublist]

                    openset_attempts = 0

                    # progress bar
                    pbar = tqdm(total=seen_dataset_size)
                    # as long as the desired generated dataset size is not reached, continue
                    while sum(class_counters) < seen_dataset_size:
                        # sample zs and classify them. Sort them according to classes
                        z_dict = sample_per_class_zs(model, num_seen_classes, batch_size, self.device,
                                                     use_new_z_bound, z_mean_bound)
                        # Calculate the distance of each z to the per class mean z.
                        z_samples_distances_to_mean = mr.calc_distances_to_means(z_means, z_dict["z_samples"],
                                                                                 self.args.distance_function)
                        # Evaluate the the statistical outlier probability using the respective Weibull model
                        z_samples_outlier_probs = mr.calc_outlier_probs(weibull_models, z_samples_distances_to_mean)

                        # For each class reject or accept the samples based on outlier probability and chosen prior.
                        for i in range(num_seen_classes):
                            # only add images if per class sample amount hasn't been surpassed yet in order
                            # to balance the dataset
                            for j in range(len(z_samples_outlier_probs[i])):
                                # check the openset threshold for each example and only add if generation is not
                                # classified as openset outlier (i.e. somewhere in "unseen" latent space)
                                if class_counters[i] < samples_per_class[i]:
                                    if z_samples_outlier_probs[i][j] < openset_threshold:
                                        zs.append(z_dict["z_samples"][i][j])
                                        targets.append(i)
                                        class_counters[i] += 1
                                        pbar.update(1)
                                else:
                                    break

                        # increment the number of open set attempts
                        openset_attempts += 1

                        # time out if none of the samples pass the above test. This can happen if either the
                        # approximate posterior has not been optimized properly and is very far away from the prior
                        # or if the rejection prior has been set to something extremely small (like 0.01%).
                        if openset_attempts == 2000 and any([val == 0 for val in class_counters]):
                            # reset samples
                            data = []
                            zs = []
                            targets = []

                            # if sampling from standard Gaussian failed the first time, try with different prior std
                            if use_new_z_bound:
                                print("\n Open set generative replay timeout")
                                openset_success = False
                                break
                            else:
                                print("\n Open set generative replay from standard Gaussian failed. Trying sampling "
                                      "with modified variance bound")
                                use_new_z_bound = True
                                openset_attempts = 0

                    pbar.close()

                    # once all the samples from the prior have been accepted to fill the entire dataset size we
                    # proceed with generating the actual data points using the probabilistic decoder.
                    if openset_success:
                        print("Openset sampling successful. Generating dataset")
                        zs = torch.stack(zs, dim=0)
                        targets = torch.LongTensor(targets)

                        # actually generate images from valid zs
                        for i in trange(0, len(zs), batch_size):
                            gen = model.module.decode(zs[i:i + batch_size])
                            gen = torch.sigmoid(gen)
                            if autoregression:
                                gen = model.module.pixelcnn.generate(gen)
                            data.append(gen.data.cpu())
                        data = torch.cat(data, dim=0)

                        # get a subset for visualization and distribute it evenly along classes
                        _, sd_idx = torch.sort(targets)
                        subset_idx = sd_idx[torch.floor(torch.arange(0, data.size(0),
                                                                     data.size(0) / self.vis_size)).long()]
                        viz_subset = data[subset_idx]

                        imgs = torchvision.utils.make_grid(viz_subset, nrow=int(math.sqrt(self.vis_size)),
                                                           padding=5)
                        torchvision.utils.save_image(viz_subset, os.path.join(save_path, 'samples_seen_tasks_' +
                                                                              str(len(self.seen_tasks) -
                                                                                  self.num_increment_tasks) + '.png'),
                                                     nrow=int(math.sqrt(self.vis_size)), padding=5)
                        writer.add_image('openset_generation_snapshot', imgs, len(self.seen_tasks) -
                                         self.num_increment_tasks)

                        # return the new trainset.
                        trainset = torch.utils.data.TensorDataset(data, targets)
                        return trainset

            # If openset generative replay with outlier rejection has failed (e.g. rejection prior set to something
            # very close to zero, model not having trained at all or approximate posterior deviating extremely from the
            # prior) or isn't desired, conventional generative replay is conducted.
            if not openset or not openset_success:
                print("Using generative model to replay old data")
                for i in trange(int(seen_dataset_size / batch_size)):
                    # sample from the prior
                    z_samples = torch.randn(batch_size, model.module.latent_dim).to(self.device)

                    # calculate probabilistic decoder, generate data points
                    gen = model.module.decode(z_samples)
                    gen = torch.sigmoid(gen)
                    if autoregression:
                        gen = model.module.pixelcnn.generate(gen)

                    # classify the samples from the prior and set the label correspondingly.
                    cl = model.module.classifier(z_samples)
                    cl = torch.nn.functional.softmax(cl, dim=1)
                    label = torch.argmax(cl, dim=1)
                    data.append(gen.data.cpu())
                    targets.append(label.data.cpu())

                # need to detach from graph as otherwise the "require grad" will crash auto differentiation
                data = torch.cat(data, dim=0)
                targets = torch.cat(targets, dim=0)

                # get a subset for visualization
                _, sd_idx = torch.sort(targets)
                subset_idx = sd_idx[torch.floor(torch.arange(0, data.size(0), data.size(0) / self.vis_size)).long()]
                viz_subset = data[subset_idx]

                imgs = torchvision.utils.make_grid(viz_subset, nrow=int(math.sqrt(self.vis_size)), padding=5)
                torchvision.utils.save_image(viz_subset, os.path.join(save_path, 'samples_seen_tasks_' +
                                                                      str(len(self.seen_tasks) -
                                                                          self.num_increment_tasks) + '.png'),
                                             nrow=int(math.sqrt(self.vis_size)), padding=5)
                writer.add_image('dataset_generation_snapshot', imgs, len(self.seen_tasks) -
                                 self.num_increment_tasks)

            # return generated trainset
            trainset = torch.utils.data.TensorDataset(data, targets)
            return trainset

    # return one of the two classes
    if args.cross_dataset:
        return CrossDataset
    else:
        return IncrementalDataset
