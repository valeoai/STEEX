import os
import sys
import argparse

from .base_options import BaseOptions


class CelebAOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(name="celeba")
        parser.set_defaults(decision_model_ckpt="celeba")

        parser.set_defaults(split="val")
        parser.set_defaults(use_ground_truth_masks=False)

        parser.set_defaults(semantic_nc=19)
        parser.set_defaults(preprocess_mode="scale_width_and_crop")
        parser.set_defaults(load_size=128)
        parser.set_defaults(crop_size=128)
        parser.set_defaults(aspect_ratio=1.0)
        parser.set_defaults(decision_model_nb_classes=3)
        parser.set_defaults(target_attribute=1) # 1 for smile, 2 for young

        return parser


class CelebAMHQOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(name="celebamaskhq")
        parser.set_defaults(decision_model_ckpt="celebamaskhq")

        parser.set_defaults(split="test")
        parser.set_defaults(use_ground_truth_masks=False)

        parser.set_defaults(semantic_nc=19)
        parser.set_defaults(preprocess_mode="scale_width_and_crop")
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(aspect_ratio=1.0)
        parser.set_defaults(decision_model_nb_classes=3)
        parser.set_defaults(target_attribute=1) # 1 for smile, 2 for young

        return parser

class BDDOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(name="bdd")
        parser.set_defaults(decision_model_ckpt="bdd")

        parser.set_defaults(split="val")
        parser.set_defaults(use_ground_truth_masks=False)

        parser.set_defaults(contain_dontcare_label=True)

        return parser


class Options(BaseOptions):
    def __init__(self,):
        pass


    def parse(self,):

        opt = BaseOptions().parse()

        # Specific parser for the specified dataset
        if opt.dataset_name == "celeba":
            parser = CelebAOptions()
        elif opt.dataset_name == "celebamhq":
            parser = CelebAMHQOptions()
        elif opt.dataset_name == "bdd":
            parser = BDDOptions()
        else:
            raise NotImplementedError

        opt = parser.parse()

        # Update paths
        if opt.dataset_name == "celeba":
            if opt.use_ground_truth_masks:
                print("No ground-truth masks for CelebA, please set --use_groun_truth_masks to False")
                assert False
            opt.image_dir = os.path.join(opt.dataroot, "celeba_squared_128", "img_squared128_celeba_%s" % split)
            opt.label_dir = os.path.join(opt.dataroot, "celeba_squared_128", "seg_squared128_celeba_%s" % split)
        elif opt.dataset_name == "celebamhq":
            mask_dir = "labels" if opt.use_ground_truth_masks else "predicted_masks"
            opt.image_dir = os.path.join(opt.dataroot, "CelebAMask-HQ", "CelebAMask-HQ", opt.split, "images")
            opt.label_dir = os.path.join(opt.dataroot, "CelebAMask-HQ", "CelebAMask-HQ", opt.split, mask_dir)
        elif opt.dataset_name == "bdd":
            mask_dir = "labels" if opt.use_ground_truth_masks else "predicted_masks"
            opt.image_dir = os.path.join(opt.dataroot, "BDD", "bdd100k", "seg", "images", opt.split)
            opt.label_dir = os.path.join(opt.dataroot, "BDD", "bdd100k", "seg", mask_dir, opt.split)
        else:
            raise NotImplementedError

        return opt


