import os
import sys
import argparse

from .base_options import BaseOptions


class CelebAOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(name="sean_celeba_gpu_2_batch_8")
        parser.set_defaults(decision_model_ckpt="decision_model_celeba")

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

        parser.set_defaults(name="CelebA-HQ_pretrained")

        parser.set_defaults(decision_model_ckpt="decision_model_celebamhq")

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

        parser.set_defaults(name="sean_bdd_gpu_4_batch_4")

        parser.set_defaults(decision_model_ckpt="decision_model_bdd")

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

        return opt


