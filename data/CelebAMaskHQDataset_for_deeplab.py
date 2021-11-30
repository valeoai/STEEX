import os
import random

import torch
import pandas as pd

from PIL import Image
from torchvision import transforms as TR


class CelebAMaskHQDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):

        opt.load_size = 256
        opt.crop_size = 256
        opt.label_nc = 18
        opt.contain_dontcare_label = True
        opt.semantic_nc = 19 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label.float()
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):

        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        image_list = pd.read_csv(os.path.join(self.opt.dataroot, 'CelebA-HQ-to-CelebA-mapping.txt'), delim_whitespace=True, header=None)

        images = []
        path_img = os.path.join(self.opt.dataroot, "CelebA-HQ-img")

        labels = []
        path_lab = os.path.join(self.opt.dataroot,"CelebAMask-HQ-mask")


        for idx,x in enumerate(image_list.loc[:,1][1:]):
            #print(idx,x)
            if mode == 'val' and int(x) >= 162771:
                images.append(str(idx)+'.jpg')
                labels.append(str(idx)+'.png')
            elif mode == 'train' and int(x) < 162771:
                images.append(str(idx)+'.jpg')
                labels.append(str(idx)+'.png')

        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace(".jpg", "") == labels[i].replace(".png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label

