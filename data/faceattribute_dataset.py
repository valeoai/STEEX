import os

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms as TR


class FaceAttributesDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, attributes_path, load_size=(256, 256), augment=False):
        super(FaceAttributesDataset, self).__init__()

        self.image_path = image_path
        self.attributes_path = attributes_path
        self.load_size = load_size
        self.augment = augment

        self.images, self.attributes,self.attributes_names = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, self.images[idx])).convert('RGB')
        attributes = self.attributes[self.images[idx]]

        image, attributes = self.transforms(image, attributes)

        return {"image": image, "attributes": attributes, "id": self.images[idx]}

    def list_images(self):

        images = []
        for item in sorted(os.listdir(self.image_path)):
            images.append(item)

        with open(self.attributes_path, "r") as f:
            lines = f.readlines()

        attributes = dict()

        attributes_names = lines[1].split(" ")

        for idx,line in enumerate(lines[2:]):
            name = line.split(" ")[0]
            attr = np.array(line.split(" ")[2:]).astype(int)
            attributes[name] = attr

        return images,attributes,attributes_names


    def transforms(self,image,attributes):

        image = TR.functional.resize(image, self.load_size, Image.BICUBIC)
        image = TR.functional.to_tensor(image)
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        attributes = torch.Tensor(attributes)
        attributes = (attributes + 1)/2

        return image, attributes

