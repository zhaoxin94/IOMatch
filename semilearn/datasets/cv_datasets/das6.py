import copy
import os

import numpy as np
from PIL import Image
from torchvision import transforms
import math

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from .datasetbase import BasicDataset

from .imagenet30 import make_dataset_from_list, pil_loader, IMG_EXTENSIONS

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def get_das6(args, alg, name, labeled_percent, num_classes, data_dir='./data'):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)),
                           int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)),
                           int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    data_dir = os.path.join(data_dir, name.lower())

    lb_dset = DAS6Dataset(root=os.path.join(data_dir, "das6"),
                          transform=transform_weak,
                          is_ulb=False,
                          alg=alg,
                          flist=os.path.join(
                              data_dir,
                              f'filelist/train_labeled_{labeled_percent}.txt'))

    ulb_dset = DAS6Dataset(root=os.path.join(data_dir, "das6"),
                           transform=transform_weak,
                           is_ulb=True,
                           alg=alg,
                           strong_transform=transform_strong,
                           flist=os.path.join(
                               data_dir, f'filelist/train_unlabeled_{labeled_percent}.txt'))

    test_dset = DAS6Dataset(root=os.path.join(data_dir, "das6"),
                            transform=transform_val,
                            is_ulb=False,
                            alg=alg,
                            flist=os.path.join(data_dir, f'filelist/test.txt'))

    test_data, test_targets = test_dset.data, test_dset.targets
    test_targets[test_targets >= num_classes] = num_classes
    seen_indices = np.where(test_targets < num_classes)[0]

    eval_dset = copy.deepcopy(test_dset)
    eval_dset.data, eval_dset.targets = eval_dset.data[
        seen_indices], eval_dset.targets[seen_indices]

    return lb_dset, ulb_dset, eval_dset, test_dset


class DAS6Dataset(BasicDataset):
    def __init__(self,
                 root,
                 transform,
                 is_ulb,
                 alg,
                 strong_transform=None,
                 flist=None):
        super(DAS6Dataset, self).__init__(alg=alg,
                                          data=None,
                                          is_ulb=is_ulb,
                                          transform=transform,
                                          strong_transform=strong_transform)
        self.root = root
        assert flist is not None, "file_list should not be None!"
        imgs, targets = make_dataset_from_list(flist)

        if len(imgs) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = pil_loader
        self.extensions = IMG_EXTENSIONS

        self.data = imgs
        self.targets = targets

        self.strong_transform = strong_transform

    def __sample__(self, idx):
        path, target = self.data[idx], self.targets[idx]
        img = self.loader(path)
        return img, target
