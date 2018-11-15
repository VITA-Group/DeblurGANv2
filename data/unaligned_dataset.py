import os
import os.path
import pathlib
import torch
import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_several
from PIL import Image
import PIL
from pdb import set_trace as st
import random
import cv2
from albumentations import Compose, Rotate, Normalize, HorizontalFlip, RandomCrop

class UnalignedDataset(BaseDataset):
    def initialize(self, config, filename):
        self.config = config
        self.filename = filename
        self.root = config['dataroot_train']

        subfolders = os.listdir(os.path.join(self.root, filename))
        subfolders_slice = subfolders
        self.dirs_A = [os.path.join(self.root, filename, subfolder, 'blur') for subfolder in subfolders_slice]

        def change_subpath(path, what_to_change, change_to):
            p = pathlib.Path(path)
            index = p.parts.index(what_to_change)
            new_path = (pathlib.Path.cwd().joinpath(*p.parts[:index])).joinpath(pathlib.Path(change_to), *p.parts[index+1:])
            return new_path

        self.A_paths = make_dataset_several(self.dirs_A)
        self.B_paths = [str(change_subpath(x, 'blur', 'sharp')) for x in self.A_paths]

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = Compose([
            HorizontalFlip(),
            Rotate(limit=20, p=0.4),
            RandomCrop(self.config['fineSize'], self.config['fineSize'])
                                  ])
        self.input_norm = Compose([Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )])

        self.output_norm = Compose([Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )])

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A_img = cv2.imread(A_path)
        A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)
        B_img = cv2.imread(B_path)
        B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=A_img, mask=B_img)

        A_img = self.input_norm(image=augmented['image'])['image']
        B_img = self.output_norm(image=augmented['mask'])['image']

        A = torch.from_numpy(np.transpose(A_img, (2, 0, 1)).astype('float32'))
        B = torch.from_numpy(np.transpose(B_img, (2, 0, 1)).astype('float32'))

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
