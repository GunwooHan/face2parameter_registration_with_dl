import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


def normalize(image):
    return (image / 127.5 - 1).astype(np.float32)


def denormalize(image):
    return ((image + 1) * 127.5)


class Normalize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super(Normalize, self).__init__(always_apply, p)
        self.norm_func = "image / 127.5 - 1"

    def apply(self, image, **params):
        return normalize(image)

    def get_transform_init_args_names(self):
        return ('norm_func', )


class DeNormalize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super(DeNormalize, self).__init__(always_apply, p)
        self.norm_func = "(image + 1) * 127.5"

    def apply(self, image, **params):
        return denormalize(image)

    def get_transform_init_args_names(self):
        return ('norm_func', )