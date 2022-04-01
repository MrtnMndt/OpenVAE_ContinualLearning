import torch
import random
import cv2
import collections
import numpy as np
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
import torchvision.transforms as transforms

"""
The code for blur augmentation is adopted in small portions from 
https://github.com/hendrycks/robustness by Hendrycks et al. 
Licensed under the Apache License, Version 2.0, with permission for commercial use, modification,
distribution, alas without liability and warranty. 
"""


def augment_flip(img, hflip_p, vflip_p):
    # horizontal flip
    rh = random.uniform(0, 1)
    # it's important that the rolled random value is smaller than the specified flip probability so that
    # when set to 0, no flips occur
    if rh < hflip_p:
        img = torch.flip(img, [2])  # flip on the width dimension as tensor is C,H,W

    # vertical flip
    rw = random.uniform(0, 1)
    if rw < vflip_p:
        img = torch.flip(img, [1])  # flip on the height dimension as tensor is C,H,W

    return img


def augment_random_translate(img, random_factor=0.1):
    random_translation_range = int(img.size(1) * random_factor)
    pad_size = random_translation_range // 2
    padded_tensor = torch.nn.functional.pad(img, (pad_size, pad_size, pad_size, pad_size))

    h, w = img.size(1), img.size(2)
    padded_h, padded_w = padded_tensor.size(1), padded_tensor.size(2)

    # minus 1 because the last value is inclusive in randint
    offset_h = random.randint(0, padded_h - h)
    offset_w = random.randint(0, padded_w - w)

    img = padded_tensor[:, offset_h:h + offset_h, offset_w:w + offset_w]

    return img


def augment_data(batch, args):
    """
    this data augmentation function is necessary because in incremental learning datasets regularly get split
    and concatenated again, which doesn't inherit the transforms. Unfortunately, the created TensorDatasets and
    ConcatDataset classes do not allow for an easy way to pass such transforms. We thus define the transforms
    by hand in this function.
    """

    for i in range(batch.size(0)):

        if args.hflip_p > 0 or args.vflip_p > 0:
            batch[i] = augment_flip(batch[i], args.hflip_p, args.vflip_p)

        if args.random_translation:
            batch[i] = augment_random_translate(batch[i])

    return batch


def blur_data(batch, patch_size, device):
    # currently no GPU implementation

    batch = batch.cpu()

    convert_img = transforms.Compose([transforms.ToPILImage()])
    convert_tensor = transforms.Compose([transforms.ToTensor()])

    blurs = get_blurs()
    random_method = random.choice(list(blurs.keys()))

    for i in range(len(batch)):
        severity = random.randint(0, 5)
        if severity > 0:
            blur = lambda clean_img: blurs[random_method](clean_img, severity, patch_size)
            batch[i] = convert_tensor(blur(convert_img(batch[i]))).float() / 255.

    return batch.to(device)


def get_blurs():
    # list of blurs to use
    d = collections.OrderedDict()
    d['Defocus Blur'] = defocus_blur
    d['Glass Blur'] = glass_blur
    d['Zoom Blur'] = zoom_blur
    d['Gaussian Blur'] = gaussian_blur
    return d


def gaussian_blur(x, severity=1, patch_size=32):
    c = [.4, .6, 0.7, .8, 1][severity - 1]

    x = gaussian(
        np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1, patch_size=32):
    # sigma, max_delta, iterations
    c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(patch_size - c[1], c[1], -1):
            for w in range(patch_size - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1, patch_size=32):
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))

    return np.clip(channels, 0, 1) * 255


def zoom_blur(x, severity=1, patch_size=32):
    def clipped_zoom(img, zoom_factor):
        h = img.shape[0]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / zoom_factor))

        top = (h - ch) // 2
        img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2

        return img[trim_top:trim_top + h, trim_top:trim_top + h]

    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)

    return np.clip(x, 0, 1) * 255
