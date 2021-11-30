import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms as TR

import data

from options.advanced_options import Options

from models.pix2pix_model import Pix2PixModel
from models.DecisionDensenetModel import DecisionDensenetModel

opt = Options().parse()

# Specify some regions for the region-targeted regime
if opt.dataset_name in ["celeba", "celebamhq"]:
    z_i_meaning = ["background", "skin", "nose", "glasses", "left_eye",
            "right_eye", "left_brow", "right_brow", "left_ear",
            "right_ear", "mouth", "upper_lip", "lower_lip", "hair",
            "hat", "earring", "necklace", "neck", "cloth", "nothing"]
    if opt.dataset_name == "celeba":
        SIZE = (128, 128)
    else:
        SIZE = (256, 256)
elif opt.dataset_name == "bdd":
    z_i_meaning = ['road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic_light', 'traffic_sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle', 'unlabeled']
    SIZE = (256, 512)
else:
    raise NotImplementedError
meaning_to_index = {meaning: i for i, meaning in enumerate(z_i_meaning)}
# Transfor the list of labels to be optimized to a set of indices
if len(opt.specified_regions) > 0:
    opt.specified_regions = set(meaning_to_index[label] for label in opt.specified_regions.split(","))

# Create experiment directories
if not os.path.exists(opt.results_dir):
    os.mkdir(opt.results_dir)
LOG_DIR = os.path.join(opt.results_dir, opt.name_exp)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
directories = {
        "query_image_dir": os.path.join(LOG_DIR, "real_images"),
        "reconstructed_image_dir": os.path.join(LOG_DIR, "reconstructed_images"),
        "counterfactual_image_dir":  os.path.join(LOG_DIR, "final_images"),
        "pkl_dir": os.path.join(LOG_DIR, "pkl_dir"),
        }
for dir_name, dir_path in directories.items():
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
# Save configuration file
config_path = os.path.join(LOG_DIR, "config.pkl")
with open(config_path, "wb") as f:
    pickle.dump(opt, f)

# Load decision model
decision_model = DecisionDensenetModel(num_classes=opt.decision_model_nb_classes, pretrained=True)
checkpoint = torch.load(os.path.join(opt.checkpoints_dir, opt.decision_model_ckpt, 'checkpoint.tar'))
decision_model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("Decision model correctly loaded. Starting from epoch", start_epoch, "with last val loss", checkpoint["loss"])
decision_model.eval().cuda()

# Load generator G
generator = Pix2PixModel(opt)
generator.eval().cuda()

## Data
dataloader = data.create_dataloader(opt)
iterable_data = iter(dataloader)

# Iterate over images
for img in range(min(len(dataloader), opt.how_many)):
    print("new batch", img, "over", opt.how_many, "batches")
    data_i = next(iterable_data)
    data_i['store_path'] = [path + "_custom" for path in data_i["path"]]

    initial_scores = decision_model(data_i['image'].cuda())
    # target = 0 if initial_score > 0.5, else target = 1
    target = (initial_scores[:, opt.target_attribute] < 0.5).double()

    # Compute reconstruction, it also generates the style codes in the folder
    reconstructed_query_image = generator(data_i, mode='inference').detach().cpu().float().numpy()

    # Get the style_codes which is the spatialized z tensor
    style_codes_numpy = np.zeros((data_i["image"].shape[0], 20, 512))
    for j, image_path in enumerate(data_i["path"]): # For each image of the batch
        img_path = os.path.split(image_path)[1]
        # loop over codes
        for i in range(20):
            style_path = os.path.join('styles_test/style_codes/', img_path, str(i), 'ACE.npy')
            if os.path.exists(style_path):
                code = np.load(style_path)
                style_codes_numpy[j, i] += code
    style_codes = torch.Tensor(style_codes_numpy).to('cuda')

    # General setting: optimize on everything
    if len(opt.specified_regions) == 0:
        z = style_codes.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=opt.lr)

    # Region-targeted setting, optimize on the labels given
    else:
        z_to_optimize_list = []
        zj_list = []
        z_list = []
        for j in range(len(style_codes)):
            for i in range(len(style_codes[j])):
                sc = style_codes[j, i].detach().clone()
                if i in opt.specified_regions:
                    sc.requires_grad = True
                    z_to_optimize_list.append(sc)
                z_list.append(sc)

        z = torch.vstack(z_list).reshape(len(style_codes), len(style_codes[0]), -1)
        optimizer = torch.optim.Adam(z_to_optimize_list, lr=opt.lr)

    # Optimization steps
    for step in range(opt.nb_steps):

        optimizer.zero_grad()

        if len(opt.specified_regions) > 0:
            z = torch.vstack(z_list).reshape(len(style_codes), len(style_codes[0]), -1)

        data_i['custom_z'] = z

        # Generate counterfactual and their proba given by the decision model
        counterfactual_image = generator(data_i, mode='inference_with_custom_z') # batch x channel x width x height
        counterfactual_logits = decision_model(counterfactual_image, before_sigmoid=True)
        counterfactual_probas = torch.sigmoid(counterfactual_logits)

        # Decision loss
        flip_decision_loss = - (1 - target) * torch.log(1 - torch.sigmoid(counterfactual_logits[:, opt.target_attribute])) - target * torch.log(torch.sigmoid(counterfactual_logits[:, opt.target_attribute]))
        loss = flip_decision_loss

        # Proximity loss
        proximity_loss = opt.lambda_prox * torch.sum(torch.square(torch.norm(z - style_codes, dim=2)), axis=1)
        loss += proximity_loss

        loss = loss.sum() # Sum over the batch
        loss.backward()

        # One optimization step
        optimizer.step()

        # Some printing
        if step % 10 == 0:
          print("Step:", step)
          print("Objective loss:", flip_decision_loss.mean().item())
          print("Difference on z:", proximity_loss.mean().item())

    # At the end, save everything
    counterfactual_image_tensor = counterfactual_image.detach().cpu().float().numpy()

    final_scores = counterfactual_probas.detach().cpu().float().numpy()
    final_loss_decision = flip_decision_loss.detach().cpu().float().numpy()
    final_loss_proximity = proximity_loss.detach().cpu().float().numpy()

    for j, image_path in enumerate(data_i["path"]):
        # Create folder for each image of the batch
        img_path = os.path.split(image_path)[1]
        path_exp = os.path.join(directories["query_image_dir"], img_path.replace('.jpg', ''))

        # Save counterfactual image
        counterfactual_image = (np.transpose(counterfactual_image_tensor[j, :, :, :], (1, 2, 0)) + 1) / 2.0 * 255.0
        counterfactual_image = counterfactual_image.astype(np.uint8)
        counterfactual_image = Image.fromarray(counterfactual_image).convert('RGB')
        counterfactual_image.save(os.path.join(directories["counterfactual_image_dir"], img_path.replace(".jpg", ".png")))

        # Save query image
        query_image = Image.open(os.path.join(opt.image_dir, img_path)).convert('RGB')
        query_image = TR.functional.resize(query_image, SIZE, Image.BICUBIC) # Resize real image
        query_image.save(os.path.join(directories["query_image_dir"], img_path.replace(".jpg", ".png")))

        # Save query reconstruction
        reconstructed_query_image_j = reconstructed_query_image[j, :, :, :]
        reconstructed_query_image_j = (np.transpose(reconstructed_query_image_j, (1, 2, 0)) + 1) / 2.0 * 255.0
        reconstructed_query_image_j = reconstructed_query_image_j.astype(np.uint8)
        reconstructed_query_image_j = Image.fromarray(reconstructed_query_image_j).convert('RGB')
        reconstructed_query_image_j.save(os.path.join(directories["reconstructed_image_dir"], img_path.replace(".jpg", ".png")))

        # Save extra stuff
        successful = np.abs(final_scores[j, opt.target_attribute] - target[j].detach().cpu().float().numpy()) < 0.5
        dump_dict = {
                  "successful": successful,
                  "initial_z": style_codes_numpy[j],
                  "final_z": z[j].detach().cpu().float().numpy(),
                  "initial_scores": initial_scores[j].detach().cpu().float().numpy(),
                  "final_scores": final_scores[j],
                  "loss_decision": final_loss_decision[j],
                  "loss_proxmity": final_loss_proximity[j],
                  }
        with open(os.path.join(directories["pkl_dir"], img_path.replace(".jpg", ".pkl")), 'wb') as f:
          pickle.dump(dump_dict, f)

