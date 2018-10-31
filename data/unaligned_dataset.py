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

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.dir_A = os.path.join(opt.dataroot, opt.phase, opt.subfolder, 'blur')
        # self.dir_B = os.path.join(opt.dataroot, opt.phase, opt.subfolder, 'sharp')

        subfolders = os.listdir(os.path.join(opt.dataroot, opt.phase))
        subfolders_slice = subfolders
        #print(subfolders)
        self.dirs_A = [os.path.join(opt.dataroot, opt.phase, subfolder, 'blur') for subfolder in subfolders_slice]
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
        #self.transform = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        B_path = self.B_paths[index % self.A_size]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        #print(A_img.size)
        w = A_img.size(2)
        h = A_img.size(1)
        # w = A_img.size[1]
        # h = A_img.size[0]
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A_img = A_img[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
        B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]




        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
