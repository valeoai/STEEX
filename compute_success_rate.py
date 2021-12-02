#!/usr/bin/env python3
import os
import pickle
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--expe', type=str, default='/path/to/experiment', help='the path of the experiment')
config=parser.parse_args()

experiment_dir = os.path.join(config.expe, "pkl_dir")

successes = []
pkl_folders = os.listdir(experiment_dir)
for folder in pkl_folders:
    # Load the pkl containing the success boolean
    with open(os.path.join(experiment_dir, folder), 'rb') as f:
        obj = pickle.load(f)
    successes.append(obj['successful'])

print("Success rate: %.3f" % np.mean(successes))
