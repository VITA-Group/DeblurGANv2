import os
import os.path
import pathlib

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
        # self.dir_A = os.path.join(opt.dataroot, opt.phase, opt.subfolder, 'blur')
        # self.dir_B = os.path.join(opt.dataroot, opt.phase, opt.subfolder, 'sharp')

        subfolders = os.listdir(os.path.join(self.root, self.config['phase']))
        subfolders_slice = subfolders
        #print(subfolders)
        self.dirs_A = [os.path.join(self.root, self.config['phase'], subfolder, 'blur') for subfolder in subfolders_slice]
        #self.dirs_B = [os.path.join(opt.dataroot, opt.phase, subfolder, 'sharp') for subfolder in subfolders_slice]
        #self.dirs_B = [x.split('/')[:-1] for x in self.dirs_A]

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)

        def change_subpath(path, what_to_change, change_to):
            p = pathlib.Path(path)
            index = p.parts.index(what_to_change)
            new_path = (pathlib.Path.cwd().joinpath(*p.parts[:index])).joinpath(pathlib.Path(change_to), *p.parts[index+1:])
            #print('new path', new_path)
            return new_path

        self.A_paths = make_dataset_several(self.dirs_A)
        #self.B_paths = make_dataset_several(self.dirs_B)
        self.B_paths = [str(change_subpath(x, 'blur', 'sharp')) for x in self.A_paths]

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = Compose([
            RandomCrop(self.config['fineSize'], self.config['fineSize']),
            HorizontalFlip(),
            Rotate(limit=20, p=0.4),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
                                  ])

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A_img = cv2.imread(A_path)
        A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)
        B_img = cv2.imread(B_path)
        B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=A_img, image2=B_img)

        A_img = augmented['image']
        B_img = augmented['image2']

        A = torch.from_numpy(np.transpose(A_img, (2, 0, 1)).astype('float32'))
        B = torch.from_numpy(np.transpose(B_img, (2, 0, 1)).astype('float32'))

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
