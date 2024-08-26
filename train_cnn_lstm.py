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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every n epochs"""
    lr = config.LR * (0.1**(epoch % config.decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_ade(outputs, targets):

    # batch_size = targets.size(0)
    outputs_ = outputs.view(outputs.size(0), -1, 2)
    targets_ = targets.view(outputs.size(0), -1, 2)
    displacement = torch.mean(torch.linalg.norm(outputs_ - targets_, dim=2), dim=1) 
    ade = torch.mean(displacement)   
    return ade

def calculate_fde(outputs, targets):

    # batch_size = targets.size(0)
    outputs_ = outputs.view(outputs.size(0), -1, 2)
    targets_ = targets.view(outputs.size(0), -1, 2)
    fde = torch.mean(torch.linalg.norm(outputs_[:, outputs_.size(1)-1, :] - targets_[:, outputs_.size(1)-1, :], dim=1))
    return fde

def evaluate(outputs, keypoints, availability):
    #denormalisation
    outputs_ = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
    keypoints_ = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)

    availability_ = availability.bool()
    outputs_ = outputs_[availability_].view(outputs.size(0), -1)
    keypoints_ = keypoints_[availability_].view(outputs.size(0), -1)

    ade = calculate_ade(outputs_, keypoints_)
    fde = calculate_fde(outputs_, keypoints_)

    return ade, fde


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

class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=512, n_layers=30):
        super(CNN_LSTM, self).__init__()

        self.resnet = resnet50(pretrained=True)

        self.num_in_channels = 3 + (2*past_trajectory)
        self.resnet.conv1 = nn.Conv2d(
            self.num_in_channels,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=False,
        )

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.num_targets = 2 * (sequence_length - past_trajectory)

        self.lstm = nn.LSTM(input_size=self.resnet.fc.in_features, hidden_size=512, num_layers=30, batch_first=True, proj_size=self.num_targets)
       
    def forward(self, x):
        batch_size = x.size(0)      
        x = self.resnet(x)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(64)
        # Passing in the input and hidden state into the model and obtaining outputs
        print(x.size())
        print(x.unsqueeze(0).size())

        out, hidden = self.lstm(x.unsqueeze(0), hidden)

        print(out.size())
        exit(0)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        # out = self.fc(out)        
        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

# # Model Creation
# def build_model() -> torch.nn.Module:
#     # load pre-trained Conv2D model
#     # '''
#     model = resnet50(pretrained=True)

#     # change input channels number to match the rasterizer's output
#     num_in_channels = 3 + (2*past_trajectory)
#     model.conv1 = nn.Conv2d(
#         num_in_channels,
#         model.conv1.out_channels,
#         kernel_size=model.conv1.kernel_size,
#         stride=model.conv1.stride,
#         padding=model.conv1.padding,
#         bias=False,
#     )
#     # change output size to (X, Y) * number of future states
#     model.fc = nn.Linear(in_features=2048, out_features=num_targets)
#     '''
#     # change input channels number to match the rasterizer's output
#     model = models.densenet161(pretrained=True)

#     num_in_channels = 3 + (2*past_trajectory)
#     model.features.conv0 = nn.Conv2d(
#         num_in_channels,
#         model.features.conv0.out_channels,
#         kernel_size=model.features.conv0.kernel_size,
#         stride=model.features.conv0.stride,
#         padding=model.features.conv0.padding,
#         bias=False,
#     )
#     # change output size to (X, Y) * number of future states
#     num_targets = 2 * (sequence_length - past_trajectory)
#     model.classifier = nn.Linear(in_features=2208, out_features=num_targets)
#     '''
#     return model

# model = build_model().to(config.DEVICE)
model = CNN_LSTM(hidden_dim=512, n_layers=30).to(config.DEVICE)
# print(model)
if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[1]).cuda()
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
    f.write("epoch, train_loss, validation_loss, learning_rate, best_epoch, best_val_loss, train_ade, train_fde, val_ade, val_fde\n")
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
    total_ade = 0.0
    total_fde = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        #print(i)
        counter += 1
        image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        outputs = outputs.view(keypoints.size(0), -1)
        availability = availability.view(availability.size(0), -1)

        if config.LOSS_FUNCTION == "ADAPTIVE":
            loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
        elif config.LOSS_FUNCTION == "MSE":
            loss = criterion(outputs, keypoints)

        if config.train_evaluate:
            ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

        loss = loss * availability

        # #newly added part including loss = loss * weights
        # av = np.array(availability.cpu())
        # weights = np.copy(av)

        # for bs_ind in range(0, weights.shape[0]):
        #     #count = (np.count_nonzero(weights[bs_ind, :] == 1, axis=0))
        #     #print(weights.shape[0])
        #     #min_weight = 1 / (2 * count)
        #     #double_weight = 2 * min_weight
        #     #triple_weight = 3 * min_weight

        #     count1 = (np.count_nonzero(weights[bs_ind, 0:40] == 1, axis=0))
        #     #count2 = (np.count_nonzero(weights[bs_ind, 20:40] == 1, axis=0))
        #     count3 = (np.count_nonzero(weights[bs_ind, 40:60] == 1, axis=0))
        #     min_weight = 1 / (count1 + (2*count3))

        #     for j_ind in range(0, 60):
        #         if weights[bs_ind, j_ind] != 0 and j_ind in range(0, 40):
        #             weights[bs_ind, j_ind] = min_weight * (count1 + count3)
        #         #elif weights[bs_ind, j_ind] != 0 and j_ind in range(20, 40):
        #             #weights[bs_ind, j_ind] = 2 * min_weight 
        #         elif weights[bs_ind, j_ind] != 0 and j_ind in range(40, 60):
        #             weights[bs_ind, j_ind] = 2 * min_weight * (count1 + count3)
        
        # weights = torch.from_numpy(weights).float().to(config.DEVICE)              
        # loss = loss * weights
        
        loss = loss.mean() 
        train_running_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        loss.backward()
        optimizer.step()

        # if tb_writer is not None:
        #     tb_writer.add_scalar('train/loss', loss.item(), epoch)
        #     if config.train_evaluate:
        #         tb_writer.add_scalar('train/ade', ade.item(), epoch)
        #         tb_writer.add_scalar('train/fde', fde.item(), epoch)
        #     tb_writer.flush()

    train_loss = train_running_loss/counter
    train_ade = total_ade/counter
    train_fde = total_fde/counter

    return train_loss, train_ade, train_fde

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    total_ade_ = 0.0
    total_fde_ = 0.0

    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            outputs = outputs.view(keypoints.size(0), -1)
            availability = availability.view(availability.size(0), -1)

            if config.LOSS_FUNCTION == "ADAPTIVE":
                loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
            elif config.LOSS_FUNCTION == "MSE":
                loss = criterion(outputs, keypoints)

            if config.train_evaluate:
                ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

            loss = loss * availability

            # #newly added part including loss = loss * weights
            # av = np.array(availability.cpu())
            # weights = np.copy(av)

            # for bs_ind in range(0, weights.shape[0]):

            #     count1 = (np.count_nonzero(weights[bs_ind, 0:40] == 1, axis=0))
            #     #count2 = (np.count_nonzero(weights[bs_ind, 20:40] == 1, axis=0))
            #     count3 = (np.count_nonzero(weights[bs_ind, 40:60] == 1, axis=0))
            #     min_weight = 1 / (count1 + (1*count3))

            #     for j_ind in range(0, 60):
            #         if weights[bs_ind, j_ind] != 0 and j_ind in range(0, 40):
            #             weights[bs_ind, j_ind] = min_weight * (count1 + count3)
            #         #elif weights[bs_ind, j_ind] != 0 and j_ind in range(20, 40):
            #             #weights[bs_ind, j_ind] = 2 * min_weight 
            #         elif weights[bs_ind, j_ind] != 0 and j_ind in range(40, 60):
            #             weights[bs_ind, j_ind] = 1 * min_weight * (count1 + count3)
            
            # weights = torch.from_numpy(weights).float().to(config.DEVICE)              
            # loss = loss * weights

            loss = loss.mean()
            valid_running_loss += loss.item()
            total_ade_ += ade.item()
            total_fde_ += fde.item()

            # if tb_writer is not None:
            #     tb_writer.add_scalar('val/loss', loss.item(), epoch)
            #     if config.train_evaluate:
            #         tb_writer.add_scalar('val/ade', ade.item(), epoch)
            #         tb_writer.add_scalar('val/fde', fde.item(), epoch)
            #     tb_writer.flush()

    valid_loss = valid_running_loss/counter
    val_ade = total_ade_/counter
    val_fde = total_fde_/counter

    return valid_loss, val_ade, val_fde

train_loss = []
val_loss = []
prev_best_val_loss = 1000000

for epoch in range(begin_epoch, config.EPOCHS):
    print(f"Epoch {epoch} of {config.EPOCHS}")

    if config.dynamic_lr:
        adjust_learning_rate(optimizer, epoch)

    train_epoch_loss, train_ade, train_fde = fit(model, train_loader, train_data, epoch)
    val_epoch_loss, val_ade, val_fde = validate(model, valid_loader, valid_data, epoch)

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
        corresponding_train_ade = train_ade
        corresponding_train_fde = train_fde
        corresponding_val_ade = val_ade
        corresponding_val_fde = val_fde

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


    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', train_epoch_loss, epoch)
        tb_writer.add_scalar('valid/loss', val_epoch_loss, epoch)
        if config.train_evaluate:
            tb_writer.add_scalar('train/ade', train_ade, epoch)
            tb_writer.add_scalar('train/fde', train_fde, epoch)
            tb_writer.add_scalar('valid/ade', val_ade, epoch)
            tb_writer.add_scalar('valid/fde', val_fde, epoch)
        tb_writer.flush()


    f.write(str(epoch)+","+str(train_epoch_loss)+","+str(val_epoch_loss)+","+str(config.LR)+","+str(best_epoch_no)+","+str(best_val_loss)+","+str(train_ade)+","+str(train_fde)+","+str(val_ade)+","+str(val_fde)+"\n")
    print(f"Train Loss: {train_epoch_loss:.8f}")
    print(f'Val Loss: {val_epoch_loss:.8f}')
    print(f'train_ade: {train_ade:.8f}')
    print(f'train_fde: {train_fde:.8f}')
    print(f'val_ade: {val_ade:.8f}')
    print(f'val_fde: {val_fde:.8f}')
    print(f'Best Val Loss: {best_val_loss:.8f}')
    print('Corresponding Train Loss:', corresponding_train_loss)
    print('Corresponding train_ade', corresponding_train_ade)
    print('Corresponding train_fde', corresponding_train_fde)
    print('Corresponding val_ade:', corresponding_val_ade)
    print('Corresponding val_fde:', corresponding_val_fde)

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
