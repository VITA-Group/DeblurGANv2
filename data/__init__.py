import os
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from torch.utils.data import Dataset


def subsample(data: Iterable[str], bounds: Tuple[float, float], n_buckets=100, salt=''):
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}'
    if salt:
        msg += f'; salt is {salt}'
    logger.info(msg)
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])


def hash_from_path(path: str, salt: str = '') -> str:
    basename = os.path.basename(path)
    return sha1(f'{basename}_{salt}'.encode()).hexdigest()


def split_into_buckets(data: Iterable[str], n_buckets, salt=''):
    hashes = map(partial(hash_from_path, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


def get_transforms(size: int, scope: str = 'train', crop='random'):
    augs = {'strong': albu.Compose([albu.HorizontalFlip(),
                                    albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                                    albu.ElasticTransform(),
                                    albu.OpticalDistortion(),
                                    albu.OneOf([
                                        albu.CLAHE(clip_limit=2),
                                        albu.IAASharpen(),
                                        albu.IAAEmboss(),
                                        albu.RandomBrightnessContrast(),
                                        albu.RandomGamma()
                                    ], p=0.5),
                                    albu.OneOf([
                                        albu.RGBShift(),
                                        albu.HueSaturationValue(),
                                    ], p=0.5),
                                    ]),
            'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size, size),
               'center': albu.CenterCrop(size, size)}[crop]
    normalize = albu.Normalize()

    pipeline = albu.Compose([aug_fn, crop_fn, normalize], additional_targets={'image2': 'image'})

    def process(a, b):
        return pipeline(a, b)

    # FixMe: make sure it applies the same ops to multiple inputs

    return process


class PairedDataset(Dataset):
    def __init__(self,
                 files_a: Tuple[str],
                 files_b: Tuple[str],
                 transform: Callable,
                 preload: bool = True,
                 preload_size: Optional[int] = 0):

        assert len(files_a) == len(files_b)

        self.preload = False
        self.data_a = files_a
        self.data_b = files_b
        self.transform = transform

        if preload:
            preload_fn = partial(self._bulk_preload, preload_size)
            self.data_a, self.data_b = (map(preload_fn, data) for data in (self.data_a, self.data_b))
            self.preload = True

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = (delayed(self._preload)(x, preload_size=preload_size) for x in data)
        return Parallel(n_jobs=cpu_count())(jobs)

    def _read_img(self, x: str):
        img = cv2.imread(x)
        assert img is not None
        return img

    def _preload(self, x: str, preload_size: int):
        img = self._read_img(x)
        h, w, *_ = img.shape
        h_scale = preload_size / h
        w_scale = preload_size / w
        scale = max(h_scale, w_scale)
        img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        assert min(img.shape[:2]) >= preload_size, f'weird img shape: {img.shape}'
        return img

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]
        if not self.preload:
            a, b = map(self._read_img, (a, b))
        a, b = self.transform(a, b)
        return {'a': a, 'b': b}

    @staticmethod
    def from_config(config):
        files_a, files_b = map(lambda x: sorted(glob(config[x], recursive=True)), ('files_a', 'files_b'))

        # ToDo: make sampling by folds
        # ToDo: choose hash function from a config

        return PairedDataset(files_a=files_a,
                             files_b=files_b,
                             preload=config['preload'],
                             preload_size=config['preload_size'],
                             transform=lambda x: x)
