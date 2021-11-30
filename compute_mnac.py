import os

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torchvision import transforms as TR

from models.OracleResnetModel import OracleResnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CHANGE HERE
checkpoint_dir = "/path/to/checkpoints/dir" # Change here to the path of checkpoints
oracle_name = "name_of_oracle" # Change here to the name fo trained oracle name (either for celeba, either for celebamhq)
name_exp = "name_exp" # replace here with real exp name

expe_folder = "results_counterfactual"
oracle_checkpoint_path = os.path.join(checkpoint_dir, oracle_name, "checkpoint.tar")
name_exp = os.path.join(expe_folder, name_exp)

oracle_pretraining_path = os.path.join(checkpoint_dir, "vggface2_pretrainings_for_oracle/resnet50_ft_dag.pth")


# Load oracle model and trained weights
model = OracleResnet(weights_path=oracle_pretraining_path, freeze_layers=True)
model.to(device)
checkpoint = torch.load(oracle_checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
lowest_loss = checkpoint['loss']
print("Checkpoint has been correctly loaded. Starting from epoch "+ str(start_epoch) +  " with last val loss " + str(lowest_loss))


total_nb_changed_attributes = 0

images = os.listdir(os.path.join(name_exp, 'real_images'))
for idx,image_name in enumerate(tqdm(images)):

    # Treat real images
    real_image = Image.open(os.path.join(name_exp,'real_images', image_name)).convert('RGB')
    real_image = TR.functional.resize(real_image, (224, 224), Image.BICUBIC)
    real_image = TR.functional.to_tensor(real_image)
    real_image = TR.functional.normalize(real_image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Treat counterfactual images
    final_image = Image.open(os.path.join(name_exp, 'final_images', image_name)).convert('RGB')
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

