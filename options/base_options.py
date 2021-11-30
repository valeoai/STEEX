import sys
import argparse

class BaseOptions():
    def __init__(self,):
        pass

    def initialize(self, parser):

        parser.add_argument('--aspect_ratio',type=float,default=2.0)
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--checkpoints_dir', type=str, help='models are saved here')
        parser.add_argument('--contain_dontcare_label',action='store_true')
        parser.add_argument('--crop_size',type=int,default=512)
        parser.add_argument('--dataset_mode',type=str,default='custom')
        parser.add_argument('--dataset_name', type=str, default="bdd")
        parser.add_argument('--decision_model',type=str,default="densenet")
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')
        parser.add_argument('--gan_mode',type=str,default='hinge')
        parser.add_argument('--gpu_ids',type=str,default='0')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--image_dir', type=str, default="path/to/image/dir")
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--instance_dir', type=str, default='')
        parser.add_argument('--isTrain',action='store_true')
        parser.add_argument('--label_dir', type=str, default='path/to/label/dir')
        parser.add_argument('--label_nc',type=int,default=19)
        parser.add_argument('--specified_regions',type=str,default="",help="list of regions to be targeted, separated by ','. Leave empty (default) to optimize over the whole image (general setting)")
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--load_size',type=int,default=512)
        parser.add_argument('--lr',type=float,default=0.01)
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--model',type=str,default='pix2pix')
        parser.add_argument('--nThreads',type=int,default=28)
        parser.add_argument('--n_layers_D',type=int,default=3)
        parser.add_argument('--name', type=str, default='sean_celeba_gpu_2_batch_8')
        parser.add_argument('--name_exp', type=str, default='default')
        parser.add_argument('--nb_steps',type=int,default=100)
        parser.add_argument('--ndf',type=int,default=64)
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--netD',type=str,default='multiscale')
        parser.add_argument('--netD_subarch',type=str,default='n_layer')
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--no_flip', type=bool, default=True)
        parser.add_argument('--no_ganFeat_loss',action='store_true')
        parser.add_argument('--no_instance', type=bool, default=True) # action='store_true', help='if specified, do *not* add instance map as input')
        parser.add_argument('--no_pairing_check',action='store_true')
        parser.add_argument('--no_vgg_loss',action='store_true')
        parser.add_argument('--norm_D',type=str,default='spectralinstance')
        parser.add_argument('--norm_E',type=str,default='spectralinstance')
        parser.add_argument('--norm_G',type=str,default='spectralspadesyncbatch3x3')
        parser.add_argument('--num_D',type=int,default=2)
        parser.add_argument('--num_upsampling_layers',type=str,default='normal')
        parser.add_argument('--output_nc',type=int,default=3)
        parser.add_argument('--patience',type=int,default=20)
        parser.add_argument('--phase',type=str,default='test')
        parser.add_argument('--preprocess_mode',type=str,default='fixed')
        parser.add_argument('--lambda_prox',type=float,default=0.3)
        parser.add_argument('--results_dir', type=str, default='results_counterfactual', help='saves results here.')
        parser.add_argument('--semantic_nc',type=int,default=20)
        parser.add_argument('--serial_batches', type=bool, default=True)
        parser.add_argument('--status',type=str,default='test')
        parser.add_argument('--target_attribute',type=int,default=0)
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
        parser.add_argument('--decision_model_ckpt', type=str)
        parser.add_argument('--decision_model_nb_classes',type=int,default=4)
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--z_dim', type=int, default=256, help="dimension of the latent z vector")

        return parser

    def parse(self,):

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()

        self.opt = opt
        return self.opt

