import os
import json

import torch
import numpy as np

from PIL import Image
from torchvision import transforms as TR


class BDDOIADataset(torch.utils.data.Dataset):
    def __init__(self, imageRoot, gtRoot, reasonRoot, cropSize=(1280, 720), augment=False):

        super(BDDOIADataset, self).__init__()

        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.reasonRoot = reasonRoot
        self.cropSize = cropSize
        self.augment = augment

        with open(gtRoot) as json_file:
            data = json.load(json_file)
        with open(reasonRoot) as json_file:
            reason = json.load(json_file)

        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = os.path.join(self.imageRoot, img['file_name'])
                if os.path.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.LongTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[ind]

        raw_image = Image.open(imgName).convert('RGB')
        target = np.array(self.targets[ind], dtype=np.int64)
        reason = np.array(self.reasons[ind], dtype=np.int64)

        image, target, reason = self.transforms(raw_image, target, reason)

        return {"image": image, "target":target, "reason":reason, "name":imgName}

    def transforms(self, raw_image, target, reason):

        if self.augment:
            pass

        new_width, new_height = (self.cropSize[1], self.cropSize[0])

        image = TR.functional.resize(raw_image, (new_width, new_height), Image.BICUBIC)
        image = TR.functional.to_tensor(image)
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        target = torch.FloatTensor(target)[0:4]
        reason = torch.FloatTensor(reason)

        return image, target, reason
