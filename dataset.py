import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CUB_dataset(Dataset):

    def __init__(self, root_path, images_path, labels, transform=None):
        self.images_path = images_path
        self.root_path = root_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path, self.images_path[idx])
        assert os.path.join(image_path), '{} image path is not exists...'.format(image_path)
        label = self.labels[idx]
        image = cv2.imread(image_path)

        if self.transform:
            image = self.transform(image)
        else:
            image = np.asarray(image)
            image = torch.from_numpy(image)

        return image, label