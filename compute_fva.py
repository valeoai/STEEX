#!/usr/bin/env python3
import os
import pickle
import argparse

import keras
import numpy as np

from PIL import Image
from tqdm import tqdm
from keras.preprocessing import image
from scipy.spatial.distance import cosine

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

parser = argparse.ArgumentParser()
parser.add_argument('--expe', type=str, default='/path/to/experiment', help='the path of the experiment')
config=parser.parse_args()

# Load the real image folder
with open(os.path.join(config.expe, "config.pkl"), "rb") as f:
    opt = pickle.load(f)
if opt.dataset_name == "celeba":
    assert opt.split == "val"
    real_images = os.path.join(opt.dataroot, "celeba_squared_128", "processed_img_squared128_celeba_val")
elif opt.dataset_name == "celebamhq":
    assert opt.split == "test"
    real_images = os.path.join(opt.dataroot, "CelebAMask-HQ", "CelebAMask-HQ", opt.split, "processed_images")
else:
    raise NotImplementedError

# Load VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

successes = []
for image_path in tqdm(os.listdir(os.path.join(config.expe, 'final_images'))):

  # Real image
  img_resized = image.load_img(os.path.join(real_images, image_path), target_size=(224, 224))
  img_numpy = image.img_to_array(img_resized)
  input_batch_real = np.expand_dims(img_numpy, axis=0)

  # Counterfactual image
  img_resized = image.load_img(os.path.join(config.expe, 'final_images', image_path), target_size=(224, 224))
  img_numpy = image.img_to_array(img_resized)
  input_batch_fake = np.expand_dims(img_numpy, axis=0)

  #Combine into a single batch
  input_batch_combined = np.concatenate((input_batch_real, input_batch_fake), axis=0)
  input_batch_combined = preprocess_input(input_batch_combined, version=2)
  preds = model.predict(input_batch_combined)

  #Cosine distance
  cosine_dist = cosine(preds[0],preds[1])

  # Sucessful is the cosine is higher than 0.5
  successes.append(cosine_dist > 0.5)

print("FVA", np.mean(successes))
