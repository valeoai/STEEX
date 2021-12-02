#!/usr/bin/env python3
import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision import transforms as TR

from models.OracleResnetModel import OracleResnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='/path/to/checkpoints', help='the path of the checkpoints')
parser.add_argument('--expe', type=str, default='/path/to/experiment', help='the path of the experiment')
config=parser.parse_args()

# Load the real image folder
with open(os.path.join(config.expe, "config.pkl"), "rb") as f:
    opt = pickle.load(f)
if opt.dataset_name == "celeba":
    assert opt.split == "val"
    real_images = os.path.join(opt.dataroot, "celeba_squared_128", "processed_img_squared128_celeba_val")
    oracle_checkpoint_path = os.path.join(config.checkpoint_dir, "oracle_attribute", "celeba", "checkpoint.tar")
elif opt.dataset_name == "celebamhq":
    assert opt.split == "test"
    real_images = os.path.join(opt.dataroot, "CelebAMask-HQ", "CelebAMask-HQ", opt.split, "processed_images")
    oracle_checkpoint_path = os.path.join(config.checkpoint_dir, "oracle_attribute", "celebamaskhq", "checkpoint.tar")
else:
    raise NotImplementedError

oracle_pretraining_path = os.path.join(config.checkpoint_dir, "vggface2_pretrainings_for_oracle/resnet50_ft_dag.pth")

# Load oracle model and trained weights
model = OracleResnet(weights_path=oracle_pretraining_path, freeze_layers=True)
model.to(device)
checkpoint = torch.load(oracle_checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
lowest_loss = checkpoint['loss']
print("Checkpoint has been correctly loaded. Starting from epoch "+ str(start_epoch) +  " with last val loss " + str(lowest_loss))


total_nb_changed_attributes = 0

images = os.listdir(os.path.join(config.expe, 'final_images'))
for idx,image_name in enumerate(tqdm(images)):

    # Treat real images
    real_image = Image.open(os.path.join(real_images, image_name)).convert('RGB')
    real_image = TR.functional.resize(real_image, (224, 224), Image.BICUBIC)
    real_image = TR.functional.to_tensor(real_image)
    real_image = TR.functional.normalize(real_image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Treat counterfactual images
    final_image = Image.open(os.path.join(config.expe, 'final_images', image_name)).convert('RGB')
    final_image = TR.functional.resize(final_image, (224, 224), Image.BICUBIC)
    final_image = TR.functional.to_tensor(final_image)
    final_image = TR.functional.normalize(final_image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Unsqueeze and put on GPU
    input_tensor = torch.cat((real_image.unsqueeze(0),final_image.unsqueeze(0)), 0).to(device)
    pred = model(input_tensor)
    pred_labels = torch.where(pred > 0.5 , 1.0, 0.0)
    diff = torch.abs(pred_labels[0] - pred_labels[1])

    nb_changed_attributes = torch.sum(diff).item()
    total_nb_changed_attributes += nb_changed_attributes

print("MNAC", total_nb_changed_attributes/len(images))

