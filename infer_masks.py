import os

import cv2
import torch
import numpy as np
import torchvision

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Args:

    checkpoints_dir = '/path/to/checkpoints/dir'

    # FOR BDD
    dataroot = '/path/to/data/BDD100k/'
    segmentation_network_name = 'deeplabv3_bdd'
    dataset_mode = 'bdd'
    save_dir_masks = '/path/to/dataset/BDD/bdd100k/seg/predicted_masks/val'
    n_classes = 20

    ## FOR CELEBAMASK-HQ
    #dataroot = '/path/to/CelebAMask-HQ/'
    #segmentation_network_name = 'deeplabv3_celebamhq'
    #dataset_mode = 'celebamhq'
    #save_dir_masks = '/path/to/dataset/CelebAMask-HQ/test/predicted_masks'
    #n_classes = 19

    seed = 42
    batch_size = 8
    no_flip = False
    phase='train'
    ckpt_iter='best'
    num_epochs=50
    pretrained = False

opt=Args()

def get_dataset_name(mode):
    if mode == "bdd":
        return "BDDDataset_for_deeplab"
    if mode == "celebamhq":
        return "CelebAMaskHQDataset_for_deeplab"
    else:
        ValueError("There is no such dataset regime as %s" % mode)

def get_dataloaders(opt):

    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("data."+dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False)

    return dataloader_train, dataloader_val

# Load validation data
_, dataloader_val = get_dataloaders(opt)

# Load trained deeplab model
deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=opt.n_classes)
checkpoint = torch.load(os.path.join(opt.checkpoints_dir, opt.segmentation_network_name, 'checkpoint.tar'))
deeplabv3.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
lowest_loss = checkpoint['loss']
print("Checkpoint has been correctly loaded. Starting from epoch", start_epoch, "with last val loss", lowest_loss)
deeplabv3.eval().cuda()

if not os.path.exists(opt.save_dir_masks):
    os.mkdir(save_dir_masks)

# Forward validation data in the deeplabv3
for data in tqdm(dataloader_val):
    inputs = data['image'].to(device)
    pred = deeplabv3(inputs)['out']
    pred_labels = pred.argmax(1)

    paths = data['name']

    for j in range(opt.batch_size):

        mask = np.asarray(pred_labels[j].cpu())

        mask = np.where(mask == 0, 256, mask)
        mask -= 1

        #print(mask)
        cv2.imwrite(os.path.join(opt.save_dir_masks, paths[j].replace('jpg', 'png')), mask)
        break

