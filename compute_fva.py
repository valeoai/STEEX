import os

import keras
import numpy as np

from PIL import Image
from tqdm import tqdm
from keras.preprocessing import image
from scipy.spatial.distance import cosine

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

# CHANGE HERE
name_exp = "name_exp" # Change to the name of experiment

expe_folder = "results_counterfactual"
name_exp = os.path.join(expe_folder, name_exp)

# Load VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

successes = []
for image_path in tqdm(os.listdir(os.path.join(name_exp, 'real_images'))):

  # Real image
  img_resized = image.load_img(os.path.join(name_exp, 'real_images', image_path), target_size=(224, 224))
  img_numpy = image.img_to_array(img_resized)
  input_batch_real = np.expand_dims(img_numpy, axis=0)

  # Counterfactual image
  img_resized = image.load_img(os.path.join(name_exp, 'final_images', image_path), target_size=(224, 224))
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
