import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor, CenterCrop, RandomRotation, RandomVerticalFlip

resolution_dict = {
    'full' : (384, 1280),
    'tu_small' : (128, 416),
    'tu_big' : (228, 912),
    'half' : (192, 640)}

class KITTIDataset(Dataset):
    def __init__(self, path, resolution=(384, 1280)):        # new augmentation='alhashim'
        self.path = path
        self.resolution = resolution
        self.files = os.listdir(self.path)

        self.transform = CenterCrop(self.resolution)

    def __getitem__(self, index):

        image_path = os.path.join(self.path, self.files[index])

        data = np.load(image_path)
        depth, image = data['depth'], data['image']

        if self.transform is not None:
            data = self.transform(data)

        image, depth = data['image'], data['depth']
        # print(image.shape)
        # print(depth.shape)

        image = np.array(image)
        depth = np.array(depth)
        return image, depth

    def __len__(self):
        return len(self.files)

