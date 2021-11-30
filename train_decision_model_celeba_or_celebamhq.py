import os

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from data.faceattribute_dataset import FaceAttributesDataset
from models.DecisionDensenetModel import DecisionDensenetModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Args:

    checkpoints_dir = "/path/to/checkopints/dir"
    data_dir = "/path/to/data/dir"

    ##################
    # FOR CELEBAMASK-HQ
    decision_model_name = 'decision_model_celebamhq'
    image_path_train = os.path.join(data_dir, "train", "images")
    image_path_val = os.path.join(data_dir, "test", "images")
    attributes_path = os.path.join(data_dir, "CelebAMask-HQ-attribute-anno.txt")
    load_size = (256, 256)

    ## FOR CELEBA
    #decision_model_name = 'decision_model_celeba'
    #image_path_train = os.path.join(data_dir, "img_squared128_celeba_train")
    #image_path_val = os.path.join(data_dir, "img_squared128_celeba_test")
    #attributes_path = os.path.join(data_dir, "list_attr_celeba.txt")
    #load_size = (128, 128)
    ###################

    train_attributes_idx = [20, 31, 39] # Male, Smile, Young
    batch_size = 32
    optimizer = 'adam'
    lr = 0.0001
    step_size = 10
    gamma_scheduler = 0.5

    num_epochs = 5

opt=Args()

# Load data
data_train = FaceAttributesDataset(image_path=opt.image_path_train, attributes_path=opt.attributes_path, load_size=opt.load_size)
data_val = FaceAttributesDataset(image_path=opt.image_path_val, attributes_path=opt.attributes_path, load_size=opt.load_size)

dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=opt.batch_size, shuffle=False, num_workers=4)

def train_one_epoch():

    print("Number of batches:", len(dataloader_train))
    total_loss = 0
    stat_loss = 0

    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    model.train()

    for batch_idx, batch_data in enumerate(tqdm(dataloader_train)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data['image']

        pred = model(inputs)
        pred_labels = torch.where(pred > 0.5, 1.0, 0.0)
        real_labels = torch.index_select(batch_data['attributes'], 1, torch.tensor(opt.train_attributes_idx).to(device))

        # Compute loss and gradients
        loss = criterion(pred, real_labels)
        acc = compute_accuracy(pred_labels, real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc

        loss.backward()
        optimizer.step()


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            log_string('mean accuracy on the last 50 batches: '+ str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss / len(dataloader_train)
    total_mean_acc = total_acc / len(dataloader_train)
    log_string('mean loss over training set: %f' % (total_mean_loss))
    log_string('mean accuracy over training set: ' + str(total_mean_acc))

    return total_mean_loss


def evaluate_one_epoch():

    model.eval()

    total_loss = 0
    stat_loss = 0
    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    print("Number of batches:", len(dataloader_val))

    for batch_idx, batch_data in enumerate(tqdm(dataloader_val)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass

        inputs = batch_data['image']
        with torch.no_grad():
            pred = model(inputs)
            pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

            real_labels = torch.index_select(batch_data['attributes'],1,torch.tensor(opt.train_attributes_idx).to(device))

        # Compute loss and metrics
        loss = criterion(pred,real_labels)
        acc = compute_accuracy(pred_labels,real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            log_string('mean accuracy on the last 50 batches: ' + str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss/len(dataloader_val)
    total_mean_acc = total_acc/len(dataloader_val)

    log_string('mean loss over validation set: %f'%(total_mean_loss))
    log_string('mean accuracy over validation set: '+str(total_mean_acc))

    return total_mean_loss

model = DecisionDensenetModel(num_classes=len(opt.train_attributes_idx), pretrained=False)
model.to(device)

if opt.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr)

criterion = nn.BCELoss(reduction='mean')

def compute_accuracy(pred, target):
    same_ids = (pred == target).float().cpu()
    return torch.mean(same_ids,axis=0).numpy()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma_scheduler, verbose=True)

LOG_DIR = os.path.join(opt.checkpoints_dir, opt.decision_model_name)
print(LOG_DIR)

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

lowest_loss = 100000
log_string("Starting training from the beginning.")

for epoch in range(opt.num_epochs):

    # Train one epoch
    log_string(' **** EPOCH: %03d ****' % (epoch+1))
    train_one_epoch()

    # Evaluate one epoch
    log_string(' **** EVALUATION AFTER EPOCH %03d ****' % (epoch+1))
    total_mean_loss = evaluate_one_epoch()
    if total_mean_loss < lowest_loss:
        lowest_loss = total_mean_loss
        save_dict = {'epoch': epoch+1, 'optimizer_state_dict': optimizer.state_dict(), 'loss': total_mean_loss, 'model_state_dict': model.state_dict()}
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

    scheduler.step()

