#!/usr/bin/env python3
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--expe', type=str, default='/path/to/experiment', help='the path of the experiment')
config=parser.parse_args()

final_images = os.path.join(config.expe, "final_images")

# Load the real image folder
with open(os.path.join(config.expe, "config.pkl"), "rb") as f:
    opt = pickle.load(f)
if opt.dataset_name == "celeba":
    assert opt.split == "val"
    real_images = os.path.join(opt.dataroot, "celeba_squared_128", "processed_img_squared128_celeba_val")
elif opt.dataset_name == "celebamhq":
    assert opt.split == "test"
    real_images = os.path.join(opt.dataroot, "CelebAMask-HQ", "CelebAMask-HQ", opt.split, "processed_images")
elif opt.dataset_name == "bdd":
    assert opt.split == "val"
    real_images = os.path.join(opt.dataroot, "BDD", "bdd100k", "seg", "images", "processed_val_images")
else:
    raise NotImplementedError

cmd = "python -m pytorch_fid %s %s" % (real_images, final_images)
os.system(cmd)
