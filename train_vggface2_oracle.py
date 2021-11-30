import os

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm

from models.OracleResnetModel import OracleResnet
from data.faceattribute_dataset import FaceAttributesDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Args:
    checkpoints_dir = "/path/to/checkpoints/dir"
    data_dir = "/path/to/data/dir"

    # FOR CELEBAMASK-HQ
    oracle_name = 'oracle_model_celebamhq'
    image_path_train = os.path.join(data_dir, "train", "images")
    image_path_val = os.path.join(data_dir, "test", "images")
    attributes_path = os.path.join(data_dir, "CelebAMask-HQ-attribute-anno.txt")

    ## FOR CELEBA
    #oracle_name = 'oracle_model_celeba'
    #image_path_train = os.path.join(data_dir, "img_squared128_celeba_train")
    #image_path_val = os.path.join(data_dir, "img_squared128_celeba_test")
    #attributes_path = os.path.join(data_dir, "list_attr_celeba.txt")

    optimizer = 'adam'
    lr = 0.0001
    step_size = 10
    gamma_scheduler = 0.5
    num_epochs = 5

    oracle_pretraining_path = os.path.join(checkpoints_dir, "vggface2_pretrainings_for_oracle/resnet50_ft_dag.pth")

opt=Args()

# Prepare data
data_train = FaceAttributesDataset(image_path=opt.image_path_train,attributes_path=opt.attributes_path,load_size=(224,224))
data_val = FaceAttributesDataset(image_path=opt.image_path_val,attributes_path=opt.attributes_path,load_size=(224,224))

dataloader_train = torch.utils.data.DataLoader(data_train,batch_size=32, shuffle=True,num_workers=4)
dataloader_val = torch.utils.data.DataLoader(data_val,batch_size=32, shuffle=False,num_workers=4)

def train_one_epoch():

    print("Number of batches:", len(dataloader_train))
    total_loss = 0
    stat_loss = 0

    total_acc = np.zeros(40)
    stat_acc = np.zeros(40)

    model.train()

    for batch_idx, batch_data in enumerate(tqdm(dataloader_train)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data['image']

        pred = model(inputs)
        pred_labels = torch.where(pred > 0.5, 1.0, 0.0)
        real_labels = torch.index_select(batch_data['attributes'], 1, torch.tensor(list(range(40))).to(device))

        # Compute loss and gradients
        loss = criterion(pred,real_labels)
        acc = compute_accuracy(pred_labels,real_labels)

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
            log_string('mean acc. across the different attributes: ' + str(np.mean(stat_acc/batch_interval)))
            stat_loss = 0
            stat_acc = 0

    total_mean_loss = total_loss / len(dataloader_train)
    total_mean_acc = total_acc / len(dataloader_train)
    log_string('mean loss over training set: %f' % (total_mean_loss))
    log_string('mean accuracy over training set: ' + str(total_mean_acc))
    log_string('mean acc. across the different attributes: '+str(np.mean(total_mean_acc)))

    return total_mean_loss

def evaluate_one_epoch():

    model.eval()

    total_loss = 0
    stat_loss = 0
    total_acc = np.zeros(40)
    stat_acc = np.zeros(40)

    print("Number of batches:", len(dataloader_val))

    for batch_idx, batch_data in enumerate(tqdm(dataloader_val)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass
        inputs = batch_data['image']
        with torch.no_grad():
            pred = model(inputs)
            pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

            real_labels = torch.index_select(batch_data['attributes'], 1, torch.tensor(list(range(40))).to(device))

        # Compute loss and metrics
        loss = criterion(pred, real_labels)
        acc = compute_accuracy(pred_labels, real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc

        batch_interval = 50
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            log_string('mean loss on the last 50 batches: %f' % (stat_loss/batch_interval))
            log_string('mean accuracy on the last 50 batches: ' + str(stat_acc/batch_interval))
            log_string('mean acc. across the different attributes: ' + str(np.mean(stat_acc/batch_interval)))
            stat_loss = 0
            stat_acc = 0

    total_mean_loss = total_loss / len(dataloader_val)
    total_mean_acc = total_acc / len(dataloader_val)

    log_string('mean loss over validation set: %f' % (total_mean_loss))
    log_string('mean accuracy over validation set: ' + str(total_mean_acc))
    log_string('mean acc. across the different attributes: ' + str(np.mean(total_mean_acc)))

    return total_mean_loss

# Load model
model = OracleResnet(weights_path=opt.oracle_pretraining_path, freeze_layers=False, unfreeze_last_block=True)
model.to(device)

# Prepare optimizer
if opt.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr)

# Prepare loss
criterion = nn.BCELoss(reduction='mean')

def compute_accuracy(pred, target):
    same_ids = (pred == target).float().cpu()
    return torch.mean(same_ids,axis=0).numpy()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma_scheduler, verbose=True)

LOG_DIR = os.path.join(opt.checkpoints_dir, opt.oracle_name)

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

lowest_loss = 100000
start_epoch=0
log_string("Starting training from the beginning.")

for epoch in range(start_epoch, opt.num_epochs):

    log_string(' **** EPOCH: %03d ****' % (epoch+1))

    train_one_epoch()

    # Evaluate
    log_string(' **** EVALUATION AFTER EPOCH %03d ****' % (epoch+1))
    total_mean_loss = evaluate_one_epoch()
    if total_mean_loss < lowest_loss:
        lowest_loss = total_mean_loss
        save_dict = {'epoch': epoch+1, 'optimizer_state_dict': optimizer.state_dict(), 'loss': total_mean_loss, 'model_state_dict': model.state_dict()}
        torch.save(save_dict, os.path.join(opt.checkpoints_dir, opt.oracle_name, 'checkpoint.tar'))

    scheduler.step()

