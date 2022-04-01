import os
import errno
import wget
import zipfile
import glob
import librosa
import scipy
import math
from tqdm import tqdm
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import scipy.io.wavfile as wavf


class MNIST:
    """
    MNIST dataset featuring gray-scale 28x28 images of
    hand-written characters belonging to ten different classes.
    Dataset implemented with torchvision.datasets.MNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        class_to_idx (dict): Defines mapping from class names to integers.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.gray_scale = args.gray_scale

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision MNIST data loader does not provide class_to_idx
        self.class_to_idx = {'0': 0,
                             '1': 1,
                             '2': 2,
                             '3': 3,
                             '4': 4,
                             '5': 5,
                             '6': 6,
                             '7': 7,
                             '8': 8,
                             '9': 9}

    def __get_transforms(self, patch_size):
        # optionally scale the images and repeat them to three channels
        # important note: these transforms will only be called once during the
        # creation of the dataset and no longer in the incremental datasets that inherit.
        # Adding data augmentation here is thus the wrong place!

        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.MNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.MNIST('datasets/MNIST/train/', train=True, transform=self.train_transforms,
                                  target_transform=None, download=True)
        valset = datasets.MNIST('datasets/MNIST/test/', train=False, transform=self.val_transforms,
                                target_transform=None, download=True)

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class FashionMNIST:
    """
    FashionMNIST dataset featuring gray-scale 28x28 images of
    Zalando clothing items belonging to ten different classes.
    Dataset implemented with torchvision.datasets.FashionMNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.gray_scale = args.gray_scale

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.class_to_idx = {'T-shirt/top': 0,
                             'Trouser': 1,
                             'Pullover': 2,
                             'Dress': 3,
                             'Coat': 4,
                             'Sandal': 5,
                             'Shirt': 6,
                             'Sneaker': 7,
                             'Bag': 8,
                             'Ankle-boot': 9}

    def __get_transforms(self, patch_size):
        # optionally scale the images and repeat to three channels
        # important note: these transforms will only be called once during the
        # creation of the dataset and no longer in the incremental datasets that inherit.
        # Adding data augmentation here is thus the wrong place!
        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.FashionMNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.FashionMNIST('datasets/FashionMNIST/train/', train=True, transform=self.train_transforms,
                                         target_transform=None, download=True)
        valset = datasets.FashionMNIST('datasets/FashionMNIST/test/', train=False, transform=self.val_transforms,
                                       target_transform=None, download=True)

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class AudioMNIST:
    """
    AudioMNIST dataset featuring gray-scale 227x227 images of
    ten spoken digits (0-9).
    https://github.com/soerenab/AudioMNIST
    Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals.
    Becker et al. arXiv:abs/1807.03418
    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.gray_scale = args.gray_scale

        self.__path = os.path.expanduser('datasets/AudioMNIST')
        self.__download()

        self.trainset, self.valset = self.get_dataset(args.patch_size)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.class_to_idx = {'0': 0,
                             '1': 1,
                             '2': 2,
                             '3': 3,
                             '4': 4,
                             '5': 5,
                             '6': 6,
                             '7': 7,
                             '8': 8,
                             '9': 9}

    def __check_exists(self):
        """
        Check if dataset has already been downloaded
        Returns:
             bool: True if downloaded dataset has been found
        """

        return os.path.exists(os.path.join(self.__path, 'train_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'train_labels_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_labels_tensor.pt'))

    def __download(self):
        """
        Downloads the AudioMNIST dataset from the web if dataset
        hasn't already been downloaded and does a spectrogram conversion.
        The latter could potentially be refactored into a separate function and conversion parameters (here hard-coded
        according to original authors) exposed to the command line parser.
        """

        if self.__check_exists():
            return

        print("Downloading AudioMNIST dataset")

        # download files
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if not os.path.exists(os.path.join(self.__path, 'AudioMNIST-master.zip')):
            url = 'https://github.com/soerenab/AudioMNIST/archive/master.zip'
            wget_data = wget.download(url, out=self.__path)

            archive = zipfile.ZipFile(wget_data)

            for file in archive.namelist():
                if file.startswith('AudioMNIST-master/data/'):
                    archive.extract(file, self.__path)

            print("Download successful")

        audio_mnist_src = os.path.join(self.__path, 'AudioMNIST-master/data/')
        data = np.array(glob.glob(os.path.join(audio_mnist_src, "**/*.wav")))

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        # first 5-cross-validation set from https://github.com/soerenab/AudioMNIST/blob/master/preprocess_data.py
        train_folders = [28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53, 36, 57, 9, 24, 37, 2,
                         8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55,
                         12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50]
        test_folders = [26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51]

        print("Converting audio to images")
        # create train and test folders and save audios as images
        for filepath in tqdm(data):
            # the last one is just a counter for repeat of each digit, e.g. say zero once, twice, third time..

            dig, vp, rep = filepath.rstrip(".wav").split("/")[-1].split("_")

            # according to https://github.com/soerenab/AudioMNIST/blob/master/preprocess_data.py
            fs, data = wavf.read(filepath)

            # resample
            data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
            # zero padding
            if len(data) > 8000:
                raise ValueError("data length cannot exceed padding length.")
            elif len(data) < 8000:
                embedded_data = np.zeros(8000)
                offset = np.random.randint(low=0, high=8000 - len(data))
                embedded_data[offset:offset + len(data)] = data
            elif len(data) == 8000:
                # nothing to do here
                embedded_data = data
                pass

            # 1. fourier transform
            # stft, with selected parameters, spectrogram will have shape (228, 230)
            f, t, zxx = scipy.signal.stft(embedded_data, 8000, nperseg=455, noverlap=420, window='hann')
            # get amplitude
            zxx = np.abs(zxx[0:227, 2:-1])

            # if not 2, then convert to decibel
            zxx = librosa.amplitude_to_db(zxx, ref=np.max)

            # normalize from range -80,0 to 0,1
            zxx = (zxx - zxx.min()) / (zxx.max() - zxx.min())

            zxx = zxx[::-1]  # reverse the order of frequencies to fit the images in the paper
            zxx = np.atleast_3d(zxx).transpose(2, 0, 1)  # reshape to (1, img_dim_h, img_dim_w)

            # decide to which list to add (train or test)
            if int(vp) in train_folders:
                train_images.append(zxx)
                train_labels.append(int(dig))
            elif int(vp) in test_folders:
                test_images.append(zxx)
                test_labels.append(int(dig))
            else:
                raise Exception('Person neither in train nor in test set!')

        train_images = torch.Tensor(train_images).float()
        train_labels = torch.Tensor(train_labels).long()
        test_images = torch.Tensor(test_images).float()
        test_labels = torch.Tensor(test_labels).long()

        torch.save(train_images, os.path.join(self.__path, 'train_images_tensor.pt'))
        torch.save(train_labels, os.path.join(self.__path, 'train_labels_tensor.pt'))
        torch.save(test_images, os.path.join(self.__path, 'test_images_tensor.pt'))
        torch.save(test_labels, os.path.join(self.__path, 'test_labels_tensor.pt'))

        print('Done!')

    def __get_audiomnist(self, path, kind='train'):
        """
        Load Audio-MNIST data
        Parameters:
            path (str): Base directory path containing .npy files for
                the Audio-MNIST dataset
            kind (str): Accepted types are 'train' and 'validation' for
                training and validation set stored in .npy files
        Returns:
            numpy.array: images, labels
        """

        images = torch.load(os.path.join(path, kind + '_images_tensor.pt'))
        labels = torch.load(os.path.join(path, kind + '_labels_tensor.pt'))

        return images, labels

    def get_dataset(self, patch_size):
        """
        Loads and wraps training and validation datasets
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        x_train, y_train = self.__get_audiomnist(self.__path, kind='train')
        x_val, y_val = self.__get_audiomnist(self.__path, kind='test')

        # up and down-sampling
        x_train = torch.nn.functional.interpolate(x_train, size=patch_size, mode='bilinear')
        x_val = torch.nn.functional.interpolate(x_val, size=patch_size, mode='bilinear')

        if not self.gray_scale:
            x_train = x_train.repeat(1, 3, 1, 1)
            x_val = x_val.repeat(1, 3, 1, 1)

        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class KMNIST:
    """
    KMNIST dataset featuring gray-scale 28x28 images of
    Japanese Kuzushiji characters belonging to ten different classes.
    Dataset implemented with torchvision.datasets.KMNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.gray_scale = args.gray_scale

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __get_transforms(self, patch_size):
        # optionally scale the images and repeat to 3 channels to compare to color images.
        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.KMNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        trainset = datasets.KMNIST('datasets/KMNIST/train/', train=True, transform=self.train_transforms,
                                   target_transform=None, download=True)
        valset = datasets.KMNIST('datasets/KMNIST/test/', train=False, transform=self.val_transforms,
                                 target_transform=None, download=True)

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR10:
    """
    CIFAR-10 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader implemented with torchvision.datasets.CIFAR10.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.gray_scale = args.gray_scale

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.class_to_idx = {'airplane': 0,
                             'automobile': 1,
                             'bird': 2,
                             'cat': 3,
                             'deer': 4,
                             'dog': 5,
                             'frog': 6,
                             'horse': 7,
                             'ship': 8,
                             'truck': 9}

    def __get_transforms(self, patch_size):
        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR10 to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR10('datasets/CIFAR10/train/', train=True, transform=self.train_transforms,
                                    target_transform=None, download=True)
        valset = datasets.CIFAR10('datasets/CIFAR10/test/', train=False, transform=self.val_transforms,
                                  target_transform=None, download=True)

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CIFAR100:
    """
    CIFAR-100 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader implemented with torchvision.datasets.CIFAR100.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 100
        self.gray_scale = args.gray_scale

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.class_to_idx = {'apples': 0,
                             'aquariumfish': 1,
                             'baby': 2,
                             'bear': 3,
                             'beaver': 4,
                             'bed': 5,
                             'bee': 6,
                             'beetle': 7,
                             'bicycle': 8,
                             'bottles': 9,
                             'bowls': 10,
                             'boy': 11,
                             'bridge': 12,
                             'bus': 13,
                             'butterfly': 14,
                             'camel': 15,
                             'cans': 16,
                             'castle': 17,
                             'caterpillar': 18,
                             'cattle': 19,
                             'chair': 20,
                             'chimpanzee': 21,
                             'clock': 22,
                             'cloud': 23,
                             'cockroach': 24,
                             'computerkeyboard': 25,
                             'couch': 26,
                             'crab': 27,
                             'crocodile': 28,
                             'cups': 29,
                             'dinosaur': 30,
                             'dolphin': 31,
                             'elephant': 32,
                             'flatfish': 33,
                             'forest': 34,
                             'fox': 35,
                             'girl': 36,
                             'hamster': 37,
                             'house': 38,
                             'kangaroo': 39,
                             'lamp': 40,
                             'lawnmower': 41,
                             'leopard': 42,
                             'lion': 43,
                             'lizard': 44,
                             'lobster': 45,
                             'man': 46,
                             'maple': 47,
                             'motorcycle': 48,
                             'mountain': 49,
                             'mouse': 50,
                             'mushrooms': 51,
                             'oak': 52,
                             'oranges': 53,
                             'orchids': 54,
                             'otter': 55,
                             'palm': 56,
                             'pears': 57,
                             'pickuptruck': 58,
                             'pine': 59,
                             'plain': 60,
                             'plates': 61,
                             'poppies': 62,
                             'porcupine': 63,
                             'possum': 64,
                             'rabbit': 65,
                             'raccoon': 66,
                             'ray': 67,
                             'road': 68,
                             'rocket': 69,
                             'roses': 70,
                             'sea': 71,
                             'seal': 72,
                             'shark': 73,
                             'shrew': 74,
                             'skunk': 75,
                             'skyscraper': 76,
                             'snail': 77,
                             'snake': 78,
                             'spider': 79,
                             'squirrel': 80,
                             'streetcar': 81,
                             'sunflowers': 82,
                             'sweetpeppers': 83,
                             'table': 84,
                             'tank': 85,
                             'telephone': 86,
                             'television': 87,
                             'tiger': 88,
                             'tractor': 89,
                             'train': 90,
                             'trout': 91,
                             'tulips': 92,
                             'turtle': 93,
                             'wardrobe': 94,
                             'whale': 95,
                             'willow': 96,
                             'wolf': 97,
                             'woman': 98,
                             'worm': 99}

    def __get_transforms(self, patch_size):
        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR100 to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.CIFAR100('datasets/CIFAR100/train/', train=True, transform=self.train_transforms,
                                     target_transform=None, download=True)
        valset = datasets.CIFAR100('datasets/CIFAR100/test/', train=False, transform=self.val_transforms,
                                   target_transform=None, download=True)

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class SVHN:
    """
    Google Street View House Numbers SVHN dataset featuring tiny 32x32 color images of
    digits belonging to 10 different classes (0-9).
    Dataloader implemented with torchvision.datasets.SVHN.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.gray_scale = args.gray_scale

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.class_to_idx = {'0': 0,
                             '1': 1,
                             '2': 2,
                             '3': 3,
                             '4': 4,
                             '5': 5,
                             '6': 6,
                             '7': 7,
                             '8': 8,
                             '9': 9}

    def __get_transforms(self, patch_size):
        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR100 to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.SVHN('datasets/SVHN/train/', split='train', transform=self.train_transforms,
                                 target_transform=None, download=True)
        valset = datasets.SVHN('datasets/SVHN/test/', split='test', transform=self.val_transforms,
                               target_transform=None, download=True)
        extraset = datasets.SVHN('datasets/SVHN/extra', split='extra', transform=self.train_transforms,
                                 target_transform=None, download=True)

        trainset = torch.utils.data.ConcatDataset([trainset, extraset])

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class Flower5:
    """
    Oxford Flower dataset .
    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 5
        self.gray_scale = args.gray_scale

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        self.class_to_idx = {'Sunflower': 0,
                             'Daisy': 1,
                             'Iris': 2,
                             'Daffodil': 3,
                             'Pansy': 4}

    def __get_transforms(self, patch_size):
        # optionally scale the images and repeat to three channels
        # important note: these transforms will only be called once during the
        # creation of the dataset and no longer in the incremental datasets that inherit.
        # Adding data augmentation here is thus the wrong place!
        resize = patch_size + int(math.ceil(patch_size * 0.1))
        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.CenterCrop(patch_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.CenterCrop(patch_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:

            train_transforms = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.CenterCrop(patch_size),
                transforms.ToTensor(),
            ])
            val_transforms = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.CenterCrop(patch_size),
                transforms.ToTensor(),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.ImageFoder to load dataset.
        Please download the dataset at https://www.robots.ox.ac.uk/~vgg/data/flowers/
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        root = '.'
        for cur_file in ['datasets', 'flower_data', '5flowers_class']:
            root = os.path.join(root, cur_file)
            if not os.path.exists(root):
                os.path.mkdir(root)

        trainset = datasets.ImageFolder(root=root + '/train/', transform=self.train_transforms,
                                        target_transform=None)
        valset = datasets.ImageFolder(root=root + '/valid/', transform=self.val_transforms,
                                      target_transform=None)

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

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader
