import os
import os.path
import pathlib
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from data.image_folder import make_dataset, make_dataset_several
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import cv2
from albumentations import *


class DeblurDataset(data.Dataset):
    def __init__(self, config, filename):
        super(data.Dataset, self).__init__()

        self.config = config
        self.filename = filename
        self.root = config['dataroot_train']

        self.A_paths, self.B_paths = self.get_paths()
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = self.get_augmentations(filename == 'train')

        self.norm = Compose([Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )])

    def get_augmentations(self, train):
        if train:
            return Compose([
                HorizontalFlip(),
                ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                OneOf([
                    CLAHE(clip_limit=2),
                    RandomContrast(),
                    RandomBrightness(),
                    RandomGamma()
                ], p=0.5),
                OneOf([
                    RGBShift(),
                    HueSaturationValue(),
                ], p=0.5),
                RandomCrop(self.config['fineSize'], self.config['fineSize'])
            ], additional_targets={'image2': 'image'})
        else:
            return Compose([
                CenterCrop(self.config['fineSize'], self.config['fineSize'])
            ], additional_targets={'image2': 'image'})

    def get_paths(self):
        subfolders = os.listdir(os.path.join(self.root, self.filename, 'blur'))
        subfolders_slice = subfolders
        self.dirs_A = [os.path.join(self.root, self.filename, 'blur', subfolder) for subfolder in subfolders_slice]

        def change_subpath(path, what_to_change, change_to):
            p = pathlib.Path(path)
            index = p.parts.index(what_to_change)
            new_path = (pathlib.Path.cwd().joinpath(*p.parts[:index])).joinpath(pathlib.Path(change_to),
                                                                                *p.parts[index + 1:])
            return new_path

        self.A_paths = make_dataset_several(self.dirs_A)
        self.B_paths = [str(change_subpath(x, 'blur', 'sharp')) for x in self.A_paths]

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        return sorted(self.A_paths), sorted(self.B_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A_img = cv2.imread(A_path)
        A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)
        B_img = cv2.imread(B_path)
        B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=A_img, image2=B_img)

        A_img = self.norm(image=augmented['image'])['image']
        B_img = self.norm(image=augmented['image2'])['image']

        A = torch.from_numpy(np.transpose(A_img, (2, 0, 1)).astype('float32'))
        B = torch.from_numpy(np.transpose(B_img, (2, 0, 1)).astype('float32'))

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'DeblurDataset'
