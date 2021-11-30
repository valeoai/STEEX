import os
import pickle

import numpy as np

expe_folder = "results_counterfactual"
name_exp = "name_exp" # replace here with real exp name

experiment_dir = os.path.join(expe_folder, name_exp, "pkl_dir")

successes = []
pkl_folders = os.listdir(experiment_dir)
for folder in pkl_folders:
    # Load the pkl containing the success boolean
    with open(os.path.join(experiment_dir, folder), 'rb') as f:
        obj = pickle.load(f)
    successes.append(obj['successful'])

print("Success rate: %.3f" % np.mean(successes))
