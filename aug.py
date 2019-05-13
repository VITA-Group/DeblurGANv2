from functools import partial
from typing import Optional

import albumentations as albu
import numpy as np
from albumentations.augmentations import functional as F


def get_transforms(size: int, scope: str = 'weak', crop='random'):
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
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    pipeline = albu.Compose([aug_fn, crop_fn, normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def _cutout(img: np.ndarray, max_num_holes=3, max_h=7, max_w=7):
    height, width, _ = img.shape
    img = img.copy()

    for _ in range(np.random.randint(1, max_num_holes)):
        y = np.random.randint(height)
        x = np.random.randint(width)

        y1 = np.clip(y - max_h // 2, 0, height)
        y2 = np.clip(y + max_h // 2, 0, height)
        x1 = np.clip(x - max_w // 2, 0, width)
        x2 = np.clip(x + max_w // 2, 0, width)

        img[y1: y2, x1: x2] = 0
    return img


def get_corrupt_function(name: Optional[str], **kwargs):
    if name is None:
        return name
    d = {'grayscale': F.to_gray,
         'cutout': _cutout,
         }
    fn = partial(d[name], **kwargs)
    return fn
