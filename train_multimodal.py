import torch
import os
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.resnet import resnet50
import torchvision.models as models

import config
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm
import utils
import robust_loss_pytorch

from torch.utils.tensorboard import SummaryWriter

matplotlib.style.use('ggplot')

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

# Model Creation
def build_model() -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet50(pretrained=True)

    # change input channels number to match the rasterizer's output
    num_in_channels = 3 + (2*past_trajectory)
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states
    num_targets = 2 * (sequence_length - past_trajectory) * int(config.multimodal)
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

model = build_model()

if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # model = torch.nn.DataParallel(model).cuda()

model.to(config.DEVICE)

parameters = get_fine_tuning_parameters(model, config.ft_portion)

if config.LOSS_FUNCTION == "ADAPTIVE":
    adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims = 60, float_dtype=np.float32, device = 'cuda:0') #num_dims=1 before

if config.LOSS_FUNCTION == "MSE":
    optimizer = optim.Adam(parameters, lr=config.LR, weight_decay=1e-3)
    criterion = nn.MSELoss(reduction="none") # criterion = nn.SmoothL1Loss()

if config.RESUME_PATH:
    print('loading checkpoint {}'.format(config.RESUME_PATH))
    checkpoint = torch.load(config.RESUME_PATH)
    begin_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config.LOSS_FUNCTION == "ADAPTIVE":
        adaptive.load_state_dict(checkpoint['adaptive_state_dict'])
        params = list(parameters) + list(adaptive.parameters())
    print('Loaded checkpoint at epoch', checkpoint['epoch'])
    f=open(f"{config.OUTPUT_PATH}/loss.txt","a")
else:
    begin_epoch = 0
    f=open(f"{config.OUTPUT_PATH}/loss.txt","w+")
    f.write("epoch, train_loss, validation_loss, learning_rate, best_epoch, best_val_loss\n")
    if config.LOSS_FUNCTION == "ADAPTIVE":
        params = list(parameters) + list(adaptive.parameters())

if config.LOSS_FUNCTION == "ADAPTIVE":
    optimizer = torch.optim.Adam(params, lr=config.LR, weight_decay=1e-3)


logger_dir = os.path.join(config.OUTPUT_PATH, 'tb_logger')
if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)
tb_writer = SummaryWriter(logger_dir)


# training function
def fit(model, dataloader, data, epoch):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        #print(i)
        counter += 1
        image, keypoints, availability, weights_final = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), data['weights_final'].to(config.DEVICE)
  
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()

        # image = image.to(f'cuda:{model.device_ids[0]}')
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        outputs = outputs.view(keypoints.size(0), -1)

        if config.LOSS_FUNCTION == "ADAPTIVE":
            loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
        elif config.LOSS_FUNCTION == "MSE":
            loss = criterion(outputs, keypoints)

        loss = loss * weights_final        
        loss = loss * availability        
        loss = loss.mean()
        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss', loss.item(), epoch)
            tb_writer.flush()

    train_loss = train_running_loss/counter

    return train_loss

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0

    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints, availability, weights_final = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), data['weights_final'].to(config.DEVICE)

            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            outputs = outputs.view(keypoints.size(0), -1)

            if config.LOSS_FUNCTION == "ADAPTIVE":
                loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
            elif config.LOSS_FUNCTION == "MSE":
                loss = criterion(outputs, keypoints)

            loss = loss * weights_final        
            loss = loss * availability
            loss = loss.mean()
            valid_running_loss += loss.item()

            if tb_writer is not None:
                tb_writer.add_scalar('val/loss', loss.item(), epoch)
                tb_writer.flush()

    valid_loss = valid_running_loss/counter
    return valid_loss

train_loss = []
val_loss = []

prev_best_val_loss = 10000

for epoch in range(begin_epoch, config.EPOCHS):
    print(f"Epoch {epoch} of {config.EPOCHS}")
    train_epoch_loss = fit(model, train_loader, train_data, epoch)
    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)

    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)

    best_val_loss = min(val_loss)
    best_epoch_no = val_loss.index(min(val_loss))

    if best_val_loss < prev_best_val_loss:
        print("saving best epoch")
        if config.LOSS_FUNCTION == "ADAPTIVE":
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'adaptive_state_dict': adaptive.state_dict(),
                    }, f"{config.OUTPUT_PATH}/best_epoch_{config.LOSS_FUNCTION}_.pth")
        elif config.LOSS_FUNCTION == "MSE":
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, f"{config.OUTPUT_PATH}/best_epoch_{config.LOSS_FUNCTION}_.pth")

        corresponding_train_loss = train_epoch_loss

    prev_best_val_loss = best_val_loss

    print("saving last epoch")
    if config.LOSS_FUNCTION == "ADAPTIVE":
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'adaptive_state_dict': adaptive.state_dict(),
                }, f"{config.OUTPUT_PATH}/last_epoch_{config.LOSS_FUNCTION}.pth")
    elif config.LOSS_FUNCTION == "MSE":
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"{config.OUTPUT_PATH}/last_epoch_{config.LOSS_FUNCTION}.pth")


    f.write(str(epoch)+","+str(train_epoch_loss)+","+str(val_epoch_loss)+","+str(config.LR)+","+str(best_epoch_no)+","+str(best_val_loss)+"\n")
    print(f"Train Loss: {train_epoch_loss:.8f}")
    print(f'Val Loss: {val_epoch_loss:.8f}')
    print(f'Best Val Loss: {best_val_loss:.8f}')
    print('Corresponding Train Loss:', corresponding_train_loss)
    print('best_epoch_no:', best_epoch_no)

    f.close()
    f=open(f"{config.OUTPUT_PATH}/loss.txt","a+")

    if epoch%config.SAVE_AFTER==0:
        print("SAVING")
        
        if config.LOSS_FUNCTION == "ADAPTIVE":
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'adaptive_state_dict': adaptive.state_dict(),
                    }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")
        elif config.LOSS_FUNCTION == "MSE":
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")


f.close()
