import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class AniData(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.img_list = []
        self.label_list = []
        self.img_list0 = glob.glob(root + '/cat/*.jpg')
        self.label_list0 = [0] * len(self.img_list0)
        self.img_list1 = glob.glob(root + '/dog/*.jpg')
        self.label_list1 = [1] * len(self.img_list1)
        self.img_list2 = glob.glob(root + '/wild/*.jpg')
        self.label_list2 = [2] * len(self.img_list2)
        self.img_list = self.img_list0 + self.img_list1 + self.img_list2
        self.label_list = self.label_list0 + self.label_list1 + self.label_list2

    def __getitem__(self, index):
        img = Image.open(self.img_list[index % len(self.img_list)])
        item = self.transform(img)
        label = self.label_list[index % len(self.label_list)]

        return item, label

    def __len__(self):     
        return len(self.img_list)
