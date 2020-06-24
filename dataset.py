import os
import cv2
import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    def __init__(self,
                 new_size=None,  
                 crop_size=None,
                 is_flip=False,
                 to_tensor=True,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        self.new_size = new_size
        self.crop_size = crop_size
        self.is_flip = is_flip
        self.to_tensor = to_tensor
        self.mean = mean
        self.std = std

        self.files = []

    def __len__(self):
        return len(self.files)
    
    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image
    
    def label_transform(self, label):
        return np.array(label).astype('int64')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=padvalue)
        return pad_image

    def rand_crop(self, image, label):
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def rand_flip(self, image, label):
        flip = np.random.choice(2) * 2 - 1
        image = image[:, ::flip, :]
        label = label[:, ::flip]
        return image, label

    def resize(self, image, label=None):
        image = cv2.resize(image, self.new_size, 
                           interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, self.new_size, 
                           interpolation = cv2.INTER_AREA)
        else:
            return image
        
        return image, label

    def gen_sample(self, image, label):
        if self.new_size:
            image, label = self.resize(image, label)

        if self.crop_size:
            image, label = self.rand_crop(image, label)
        
        if self.is_flip:
            image, label = self.rand_flip(image, label)

        image = self.input_transform(image).transpose((2, 0, 1))
        label = self.label_transform(label)

        if self.to_tensor:
            image, label = torch.from_numpy(image), torch.from_numpy(label)

        return image, label

class BDD(BaseDataset):
    def __init__(self, 
                 root=None, 
                 mode=None, 
                 num_classes=8,
                 new_size=None,
                 crop_size=None,
                 is_flip=False,
                 to_tensor=True,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(BDD, self).__init__(new_size, crop_size, 
                                  is_flip, to_tensor, mean, std,)

        self.root = root
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_label = num_classes + 1

        if root is not None and node is not None:
            self.files = self.read_files()

        self.color_mapping = {
            0: (128, 64,128),    # road
            1: (244, 35,232),    # sidewalk
            2: ( 70, 70, 70),    # building
            3: (102,102,156),    # wall
            4: (190,153,153),    # fence
            5: (153,153,153),    # pole
            6: (250,170, 30),    # traffic light
            7: (220,220,  0),    # traffic sign
            8: (107,142, 35),    # vegetation
            9: (152,251,152),    # terrain
            10: ( 70,130,180),   # sky
            11: (220, 20, 60),   # person
            12: (255,  0,  0),   # rider
            13: (  0,  0,142),   # car
            14: (  0,  0, 70),   # truck
            15: (  0, 60,100),   # bus
            16: (  0, 80,100),   # train
            17: (  0,  0,230),   # motorcycle
            18: (119, 11, 32),   # bicycle
            255: (  0,  0,  0)   # don't care
        }

        self.label_mapping = {
            0: 0,                    # road
            1: 1,                    # sidewalk
            2: 2,                    # building
            3: self.num_classes,     # wall
            4: self.num_classes,     # fence
            5: self.num_classes,     # pole
            6: self.num_classes,     # traffic light
            7: self.num_classes,     # traffic sign
            8: 3,                    # vegetation
            9: 4,                    # terrain
            10: 5,                   # sky
            11: 6,                   # person
            12: self.num_classes,    # rider
            13: 7,                   # car
            14: self.num_classes,    # truck
            15: self.num_classes,    # bus
            16: self.num_classes,    # train
            17: self.num_classes,    # motorcycle
            18: self.num_classes,    # bicycle
            255: self.num_classes    # don't care
        }

        """self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()"""
    
    def read_files(self):
        files = []
        if 'test' in self.mode:
            imgs = sorted(glob.glob(os.path.join(self.root, 'images', self.mode, '*')))
            for img_path in imgs:
                files.append({
                    'images': img_path
                })
        else:
            imgs = sorted(glob.glob(os.path.join(self.root, 'images', self.mode, '*')))
            labs = sorted(glob.glob(os.path.join(self.root, 'labels', self.mode, '*')))
            for (img_path, lab_path) in zip(imgs, labs):
                files.append({
                    'image': img_path,
                    'label': lab_path
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    
    def label_to_color(self, label):
        label = self.convert_label(label, True)
        color = np.zeros((*label.shape, 3), dtype=np.uint8)
        for k, v in self.color_mapping.items():
            color[label == k] = v
        return color

    def __getitem__(self,index):
        item = self.files[index]
        image = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.mode:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            return image

        label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label)
        
        return image, label
