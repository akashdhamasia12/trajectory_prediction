import torch
import random
import os
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from torchvision.models.resnet import resnet50
from resnet50_dropout import resnet50
import torchvision.models as models

import config

if config.DATASET == "SHIFT":
    from dataset_shift import train_data, train_loader, valid_data, valid_loader, valid_loader_out
else:
    from dataset import train_data, train_loader, valid_data, valid_loader, valid_loader_noise

from tqdm import tqdm
import utils
import robust_loss_pytorch

import copy
# from torch.utils.tensorboard import SummaryWriter

matplotlib.style.use('ggplot')

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY

#set random seeds
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
print("Random seed", config.seed)

learning_rate = config.LR

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every n epochs"""
    lr = config.LR * (0.1**(epoch % config.decay_rate))
    learning_rate = lr
    if lr < config.min_lr:
        lr = config.min_lr

    # lr = 0.01
    # if epoch == 10:
    #     lr = 0.001    
    # if epoch == 50:
    #     lr = 0.0001
    # if epoch == 80:
    #     lr = 0.001

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
    # outputs_ = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
    # keypoints_ = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)

    availability_ = availability.bool()
    outputs_ = outputs[availability_].view(outputs.size(0), -1)
    keypoints_ = keypoints[availability_].view(outputs.size(0), -1)

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

def set_training_mode_for_dropout(net, training=True):
    """Set Dropout mode to train or eval."""

    for m in net.modules():
#        print(m.__class__.__name__)
        if m.__class__.__name__.startswith('Dropout'):
            if training==True:
                m.train()
            else:
                m.eval()
    return net        


def get_EAUC_loss(ade_all, ave_all): 

        robust = torch.div(ade_all, 30)
        conf = torch.div(ave_all, 3)

        #New ADEvsUC Loss --> Current version is for each sample #ade_th was 0.7
        # ade_th = torch.tensor(0.8, device=ade_all.device) #the threshold value will be obtained from first training epochs
        # u_th = torch.tensor(0.6, device=ade_all.device) #the threshold value will be obtained from first training epochs
        # eps=1e-10
        # beta=200 #20 #0.001 Birinci testteki 20'li

        ade_th = torch.tensor(0.2, device=ade_all.device) #the threshold value will be obtained from first training epochs
        u_th = torch.tensor(0.3, device=ade_all.device) #the threshold value will be obtained from first training epochs
        # eps=1e-10
        beta=200 #20 #0.001 Birinci testteki 20'li

        eps = torch.tensor(1e-10, device=ade_all.device)

        robust = torch.where(robust > 1, 1 - eps, robust)
        conf = torch.where(conf > 1, 1 - eps, conf)

        n_lc = torch.zeros(1, device=ade_all.device)  # number of samples accurate and certain
        n_hc = torch.zeros(1, device=ade_all.device)  # number of samples inaccurate and certain
        n_lu = torch.zeros(1, device=ade_all.device)  # number of samples accurate and uncertain
        n_hu = torch.zeros(1, device=ade_all.device)  # number of samples inaccurate and uncertain

        for i in range(len(ade_all)):

            if ((robust[i].item() <= ade_th.item())
                    and conf[i].item() < u_th.item()):
                """ low mse and certain """
                n_lc += (1 - torch.tanh((robust[i]))) * (1 - torch.tanh((conf[i])))
            elif ((robust[i].item() <= ade_th.item())
                    and conf[i].item() >= u_th.item()):
                """ low mse and uncertain """
                n_lu += (1 - torch.tanh(robust[i])) * (torch.tanh(conf[i]))
            elif ((robust[i].item() > ade_th.item())
                    and conf[i].item() < u_th.item()):
                """ high mse and certain """
                n_hc += (torch.tanh(robust[i])) * (1 - torch.tanh((conf[i])))
            elif (((robust[i].item() > ade_th.item()))
                    and conf[i].item() >= u_th.item()):
                """ high mse and uncertain """
                n_hu += (torch.tanh(robust[i])) * (torch.tanh(conf[i]))

        avu = ((3*n_lc) + n_hu) / ((3*n_lc) + n_lu + n_hc + n_hu + eps)
        #p_lc = (n_lc) / (n_lc + n_hc)
        #p_hu = (n_hu) / (n_hu + n_hc)
        # print('avu:', avu, 'n_lc:', n_lc, 'n_hu:', n_hu, 'n_lu:', n_lu, 'n_hc:', n_hc, '\n')
        #print('Actual AvU: ', accuracy_vs_uncertainty(ade_all, ave_all,
                            #ade_th, u_th))
        avu_loss = -1 * beta * torch.log(avu + eps)

        return avu_loss


# Model Creation
def build_model() -> torch.nn.Module:
    # load pre-trained Conv2D model
    # '''
    model = resnet50(pretrained=True, p=config.dropout_prob)

    # change input channels number to match the rasterizer's output
    if config.DATASET == "SHIFT":
        num_in_channels = 25
    else:
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

    if config.future_prediction > 0:
        num_targets = 2 * config.future_prediction
    else:
        num_targets = 2 * (sequence_length - past_trajectory)

    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

model = build_model().to(config.DEVICE)
# print(model)
if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[config.cuda_device]).cuda()
    # model = torch.nn.DataParallel(model).cuda()
model.to(config.DEVICE)

parameters = get_fine_tuning_parameters(model, config.ft_portion)

if config.LOSS_FUNCTION == "ADAPTIVE":
    adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims = 60, float_dtype=np.float32, device = 'cuda:0') #num_dims=1 before

if config.LOSS_FUNCTION == "MSE":
    optimizer = optim.Adam(parameters, lr=config.LR, weight_decay=1e-4)
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
    f.write("epoch, train_loss, validation_loss, learning_rate, best_epoch, best_val_loss, train_ade, train_fde, val_ade, val_fde, val_ade_out, val_fde_out, train_uncertainty, val_uncertainty, val_uncertainty_out\n")
    if config.LOSS_FUNCTION == "ADAPTIVE":
        params = list(parameters) + list(adaptive.parameters())

if config.LOSS_FUNCTION == "ADAPTIVE":
    optimizer = torch.optim.Adam(params, lr=config.LR, weight_decay=1e-3)


# logger_dir = os.path.join(config.OUTPUT_PATH, 'tb_logger')
# if not os.path.exists(logger_dir):
#     os.makedirs(logger_dir)
# tb_writer = SummaryWriter(logger_dir)
# training function
def fit(model, dataloader, data, epoch):
    print('Training')
    model.train()
    train_running_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    counter = 0
    uncertainty_var = 0
    eauc_loss = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        #print(i)
        counter += 1
        image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        availability = availability.view(availability.size(0), -1)
        optimizer.zero_grad()

        #denormalization
        keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)            

        for mc in range(0, config.num_monte_carlo_training):
            outputs, _ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping    
            #denormalization
            outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)            
            outputs = outputs.view(keypoints.size(0), -1)
            outputs = torch.unsqueeze(outputs, 0)
            if mc == 0:
                outputs_list = outputs
            else:
                outputs_list = torch.cat((outputs_list, outputs), 0)

        # outputs = torch.mean(outputs_list, 0)
        outputs = torch.squeeze(outputs_list[0], 0)

        if config.num_monte_carlo_training > 1:
            #denormalisation
            # outputs_list = ((outputs_list + 1)/2)*int(config.IMAGE_SIZE)            
            outputs_var = torch.std(outputs_list, 0)
            outputs_var_ = outputs_var.view(outputs_var.size(0), -1, 2)
            outputs_var_mean = torch.mean(outputs_var_, 2)
            total_uncertainty_ = torch.mean(outputs_var_mean, 1)
            total_uncertainty = torch.mean(total_uncertainty_)

        if config.LOSS_FUNCTION == "ADAPTIVE":
            loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
        elif config.LOSS_FUNCTION == "MSE":
            loss = criterion(outputs, keypoints)

        if config.quantile_regression == True:
            loss = torch.max(config.quantile*loss, (config.quantile-1)*loss)

        if config.train_evaluate:
            ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

        if config.num_monte_carlo_training > 1:
        #     loss = loss + outputs_var*config.uncertainty_factor
            EaUC = get_EAUC_loss(torch.mean(loss, 1), total_uncertainty_)
            # print(EaUC)

        
        loss = loss * availability        
        loss = loss.mean()

        if config.num_monte_carlo_training > 1:
            loss = loss + EaUC

        train_running_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        if config.num_monte_carlo_training > 1:
            uncertainty_var += total_uncertainty.item()    
            eauc_loss += EaUC.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss/counter
    train_ade = total_ade/counter
    train_fde = total_fde/counter
    train_uncertainty = uncertainty_var/counter
    train_eauc = eauc_loss/counter

    return train_loss, train_ade, train_fde, train_uncertainty, train_eauc

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    # model.eval()
    valid_running_loss = 0.0
    total_ade_ = 0.0
    total_fde_ = 0.0
    uncertainty_var = 0
    eauc_loss = 0


    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            # outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            # outputs = outputs.view(keypoints.size(0), -1)
            availability = availability.view(availability.size(0), -1)

            #denormalization
            keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)            

            for mc in range(0, config.num_monte_carlo_training):
                model.eval()
                model = set_training_mode_for_dropout(model, True)
                outputs, _ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping    
                #denormalization
                outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)            
                outputs = outputs.view(keypoints.size(0), -1)
                outputs = torch.unsqueeze(outputs, 0)
                if mc == 0:
                    outputs_list = outputs
                else:
                    outputs_list = torch.cat((outputs_list, outputs), 0)
                model = set_training_mode_for_dropout(model, False)

            # outputs = torch.mean(outputs_list, 0)
            outputs = torch.squeeze(outputs_list[0], 0)

            if config.num_monte_carlo_training > 1:
                #denormalisation
                # outputs_list = ((outputs_list + 1)/2)*int(config.IMAGE_SIZE)            
                outputs_var = torch.std(outputs_list, 0)
                outputs_var_ = outputs_var.view(outputs_var.size(0), -1, 2)
                outputs_var_mean = torch.mean(outputs_var_, 2)
                total_uncertainty_ = torch.mean(outputs_var_mean, 1)
                total_uncertainty = torch.mean(total_uncertainty_)

            if config.LOSS_FUNCTION == "ADAPTIVE":
                loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
            elif config.LOSS_FUNCTION == "MSE":
                loss = criterion(outputs, keypoints)

            if config.quantile_regression == True:
                loss = torch.max(config.quantile*loss, (config.quantile-1)*loss)

            if config.train_evaluate:
                ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

            if config.num_monte_carlo_training > 1:
            #     loss = loss + outputs_var*config.uncertainty_factor
                EaUC = get_EAUC_loss(torch.mean(loss, 1), total_uncertainty_)

            loss = loss * availability
            loss = loss.mean()

            if config.num_monte_carlo_training > 1:
                loss = loss + EaUC

            valid_running_loss += loss.item()
            total_ade_ += ade.item()
            total_fde_ += fde.item()
            if config.num_monte_carlo_training > 1:
                uncertainty_var += total_uncertainty.item()    
                eauc_loss += EaUC.item()

    valid_loss = valid_running_loss/counter
    val_ade = total_ade_/counter
    val_fde = total_fde_/counter
    val_uncertainty = uncertainty_var/counter
    val_eauc = eauc_loss/counter

    return valid_loss, val_ade, val_fde, val_uncertainty, val_eauc

train_loss = []
val_loss = []
prev_best_val_loss = 1000000
val_ade_out = -999
val_fde_out = -999
val_uncertainty_out = -999

for epoch in range(begin_epoch, config.EPOCHS):
    print(f"Epoch {epoch} of {config.EPOCHS}")

    if config.dynamic_lr:
        adjust_learning_rate(optimizer, epoch)

    train_epoch_loss, train_ade, train_fde, train_uncertainty, train_eauc = fit(model, train_loader, train_data, epoch)
    val_epoch_loss, val_ade, val_fde, val_uncertainty, val_eauc = validate(model, valid_loader, valid_data, epoch)
    if config.DATASET == "SHIFT":
        val_epoch_loss_out, val_ade_out, val_fde_out, val_uncertainty_out = validate(model, valid_loader_out, valid_data, epoch)
    if config.noise == True:
        val_epoch_loss_out, val_ade_out, val_fde_out, val_uncertainty_out = validate(model, valid_loader_noise, valid_data, epoch)

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
        corresponding_val_ade_out = val_ade_out
        corresponding_val_fde_out = val_fde_out
        corresponding_train_uncertainty = train_uncertainty
        corresponding_val_uncertainity = val_uncertainty
        corresponding_val_uncertainity_out = val_uncertainty_out
        corresponding_train_eauc = train_eauc
        corresponding_val_eauc = val_eauc

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


    f.write(str(epoch)+","+str(train_epoch_loss)+","+str(val_epoch_loss)+","+str(learning_rate)+","+str(best_epoch_no)+","+str(best_val_loss)+","+str(train_ade)+","+str(train_fde)+","+str(val_ade)+","+str(val_fde)+","+str(val_ade_out)+","+str(val_fde_out)+","+str(train_uncertainty)+","+str(val_uncertainty)+","+str(val_uncertainty_out)+","+str(train_eauc)+","+str(val_eauc)+"\n")
    print(f"Train Loss: {train_epoch_loss:.8f}")
    print(f'Val Loss: {val_epoch_loss:.8f}')
    print(f'train_ade: {train_ade:.8f}')
    print(f'train_fde: {train_fde:.8f}')
    print(f'val_ade: {val_ade:.8f}')
    print(f'val_fde: {val_fde:.8f}')
    print(f'train_uncertainty: {train_uncertainty:.8f}')
    print(f'val_uncertainty: {val_uncertainty:.8f}')
    print(f'val_uncertainty_out: {val_uncertainty_out:.8f}')
    print(f'Best Val Loss: {best_val_loss:.8f}')
    print(f'train_eauc: {train_eauc:.8f}')
    print(f'val_eauc: {val_eauc:.8f}')
    print('Corresponding Train Loss:', corresponding_train_loss)
    print('Corresponding train_ade', corresponding_train_ade)
    print('Corresponding train_fde', corresponding_train_fde)
    print('Corresponding val_ade:', corresponding_val_ade)
    print('Corresponding val_fde:', corresponding_val_fde)
    print('Corresponding val_ade_out:', corresponding_val_ade_out)
    print('Corresponding val_fde_out:', corresponding_val_fde_out)
    print('Corresponding_train_uncertainty:', corresponding_train_uncertainty)
    print('Corresponding_val_uncertainity:', corresponding_val_uncertainity)
    print('Corresponding_val_uncertainity_out:', corresponding_val_uncertainity_out)
    print('Corresponding_train_eauc:', corresponding_train_eauc)
    print('Corresponding_val_eauc:', corresponding_val_eauc)

    print('best_epoch_no:', best_epoch_no)

    f.close()
    f=open(f"{config.OUTPUT_PATH}/loss.txt","a+")

    # if epoch%config.SAVE_AFTER==0:
    #     print("SAVING")
        
    #     if config.LOSS_FUNCTION == "ADAPTIVE":
    #         torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'adaptive_state_dict': adaptive.state_dict(),
    #                 }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")
    #     elif config.LOSS_FUNCTION == "MSE":
    #         torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': criterion,
    #                 }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")


f.close()
