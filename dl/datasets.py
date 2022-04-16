import os
import glob
import random

import numpy as np
import cv2
import torch


class P2FRegDataset(torch.utils.data.Dataset):
    def __init__(self, images, target_transform=None, source_transform=None):
        self.images = images
        self.target_transform = target_transform
        self.source_transform = source_transform
        self.count = 30


    def __len__(self):
        return len(self.images)

    def __getitem__(self, target_idx):
        if self.count % 5 == 0:
            source_idx = random.randint(0, len(self.images) - 1)
            target_idx = source_idx
            self.count = 1
        else:
            source_idx = random.randint(0, len(self.images) - 1)
            while target_idx == source_idx:
                source_idx = random.randint(0, len(self.images) - 1)

            self.count += 1

        target_image = cv2.imread(self.images[target_idx])
        source_image = cv2.imread(self.images[source_idx])

        if self.target_transform:
            target_image = self.target_transform(target_image)
        if self.source_transform:
            source_image = self.source_transform(source_image)

        return target_image, source_image


if __name__ == '__main__':
    import pandas as pd

