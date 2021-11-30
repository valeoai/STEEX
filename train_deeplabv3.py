import os

import numpy as np
import torchvision
import torch
import torch.nn as nn

from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Args:

    checkpoints_dir = '/path/to/checkpoints/dir'

    # FOR BDD
    dataroot = '/path/to/bdd_dataset/'
    segmentation_network_name = 'deeplabv3_bdd'
    dataset_mode = 'bdd'
    n_classes = 20

    ## FOR CELEBAMASK-HQ
    #dataroot = '/path/to/CelebAMask-HQ/'
    #segmentation_network_name = 'deeplabv3_celebamhq'
    #dataset_mode = 'celebamhq'
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

def compute_iou(pred, target, n_classes = opt.n_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds * target_inds).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)

def compute_accuracy(pred, target, n_classes=opt.n_classes):
    accs = []
    pred = pred.view(-1)
    target = target.view(-1)
    same_ids = pred == target
    return same_ids.long().sum().data.cpu().item()/float(torch.numel(pred))

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=opt.pretrained, num_classes=opt.n_classes)
model.train().to(device)

# Data
dataloader_train, dataloader_val = get_dataloaders(opt)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss(ignore_index=0)

## Training loop
def train_one_epoch():

    print("Number of batches:", len(dataloader_train))
    total_loss = 0
    stat_loss = 0
    total_iou = 0
    stat_iou = 0
    total_acc = 0
    stat_acc = 0
    model.train()

    for batch_idx, batch_data in enumerate(tqdm(dataloader_train)):
        batch_data['image'] = batch_data['image'].to(device)
        #print(batch_data['label'].shape)
        batch_data['label'] = batch_data['label'].squeeze(1).long().to(device)
        #print(batch_data['label'].shape)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data['image']

        pred = model(inputs)['out']
        pred_labels = pred.argmax(1)


        # Compute loss and gradients
        loss = criterion(pred,batch_data['label'])
        loss.backward()
        optimizer.step()

        iou = np.nanmean(compute_iou(pred_labels,batch_data['label']))
        acc = compute_accuracy(pred_labels,batch_data['label'])

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_iou += iou.item()
        total_iou += iou.item()
        stat_acc += acc
        total_acc += acc


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            log_string('mean IoU on the last 50 batches: %f'%(stat_iou/batch_interval))
            log_string('mean pixel accuracy on the last 50 batches: %f'%(stat_acc/batch_interval))
            stat_loss = 0
            stat_iou = 0
            stat_acc = 0

    total_mean_loss = total_loss/len(dataloader_train)
    total_mean_iou = total_iou/len(dataloader_train)
    total_mean_acc = total_acc/len(dataloader_train)
    log_string('mean loss over training set: %f'%(total_mean_loss))
    log_string('mean IoU over training set: %f'%(total_mean_iou))
    log_string('mean pixel accuracy over training set: %f'%(total_mean_acc))

    return total_mean_loss


def evaluate_one_epoch():

    model.eval()

    total_loss = 0
    stat_loss = 0
    total_iou = 0
    stat_iou = 0
    total_acc = 0
    stat_acc = 0

    print("Number of batches:", len(dataloader_val))
    for batch_idx, batch_data in enumerate(tqdm(dataloader_val)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['label'] = batch_data['label'].squeeze(1).long().to(device)

        # Forward pass

        inputs = batch_data['image']
        with torch.no_grad():
            pred = model(inputs)['out']
            pred_labels = pred.argmax(1)

        # Compute loss and metrics
        loss = criterion(pred,batch_data['label'])
        iou = np.nanmean(compute_iou(pred_labels,batch_data['label']))
        acc = compute_accuracy(pred_labels,batch_data['label'])


        stat_loss += loss.item()
        total_loss += loss.item()
        stat_iou += iou.item()
        total_iou += iou.item()
        stat_acc += acc
        total_acc += acc

        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            log_string('mean IoU on the last 50 batches: %f'%(stat_iou/batch_interval))
            log_string('mean pixel accuracy on the last 50 batches: %f'%(stat_acc/batch_interval))
            stat_loss = 0
            stat_iou = 0
            stat_acc = 0

    total_mean_loss = total_loss/len(dataloader_val)
    total_mean_iou = total_iou/len(dataloader_val)
    total_mean_acc = total_acc/len(dataloader_val)
    log_string('mean loss over validation set: %f'%(total_mean_loss))
    log_string('mean IoU over validation set: %f'%(total_mean_iou))
    log_string('mean pixel accuracy over training set: %f'%(total_mean_acc))
    return total_mean_loss

LOG_DIR	= os.path.join(opt.checkpoints_dir, opt.segmentation_network_name)

if not os.path.exists(LOG_DIR):
	os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

lowest_loss = 100000
for epoch in range(opt.num_epochs):

    # Train one epoch
    print(' **** EPOCH: %03d ****' % (epoch+1))
    train_one_epoch()

    # Periodically evaluate
    if epoch == 0 or epoch % 5 == 4:
        print(' **** EVALUATION AFTER EPOCH %03d ****' % (epoch+1))
        total_mean_loss = evaluate_one_epoch()
        if total_mean_loss < lowest_loss:
            lowest_loss = total_mean_loss
            save_dict = {'epoch': epoch+1, 'optimizer_state_dict': optimizer.state_dict(), 'loss': total_mean_loss, 'model_state_dict': model.state_dict()}
            torch.save(save_dict, os.path.join(opt.checkpoints_dir, opt.segmentation_network_name, 'checkpoint.tar'))
