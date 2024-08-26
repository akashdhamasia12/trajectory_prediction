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

from bayesian_torch_.bayesian_torch.models.bayesian import resnet_variational_large as resnet
from bayesian_torch_.bayesian_torch.models.deterministic import resnet_large as det_resnet
from bayesian_torch_.bayesian_torch.utils.util import get_rho

from bayesian_torch_.bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch_.bayesian_torch.layers import LinearReparameterization

from torch.utils.tensorboard import SummaryWriter

matplotlib.style.use('ggplot')

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
prior_mu = config.prior_mu_
prior_sigma = config.prior_sigma_
posterior_mu_init = config.posterior_mu_init_
posterior_rho_init = config.posterior_rho_init_

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

def MOPED_layer(layer, det_layer, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)

    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes. AAAI 2020.
    """

    if (str(layer) == 'Conv2dReparameterization()'):
        #set the priors
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize surrogate posteriors
        layer.mu_kernel.data = det_layer.weight.data
        layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif (isinstance(layer, nn.Conv2d)):
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data2

    elif (str(layer) == 'LinearReparameterization()'):
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize the surrogate posteriors

        layer.mu_weight.data = det_layer.weight.data
        layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif str(layer).startswith('Batch'):
        #initialize parameters
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data
        layer.running_mean.data = det_layer.running_mean.data
        layer.running_var.data = det_layer.running_var.data
        layer.num_batches_tracked.data = det_layer.num_batches_tracked.data


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
def build_model():

    num_in_channels = 3 + (2*past_trajectory)
    model = resnet.__dict__["resnet50"]()
    model.conv1 = Conv2dReparameterization(
        in_channels=num_in_channels,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        prior_mean=prior_mu,
        prior_variance=prior_sigma,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init,
        bias=False)

    # change output size to (X, Y) * number of future states
    num_targets = 2 * (sequence_length - past_trajectory)
    model.fc = LinearReparameterization(
        in_features=model.fc.in_features,
        out_features=num_targets,
        prior_mean=prior_mu,
        prior_variance=prior_sigma,
        posterior_mu_init=posterior_mu_init,
        posterior_rho_init=posterior_rho_init)

    # det_model = det_resnet.__dict__["resnet50"](pretrained=True)
    # det_model.conv1 = nn.Conv2d(in_channels=num_in_channels,
    #                         out_channels=64,
    #                         kernel_size=7,
    #                         stride=2,
    #                         padding=3,
    #                         bias=False)

    # det_model.fc = nn.Linear(det_model.fc.in_features, num_targets)


    # return model.to(config.DEVICE), det_model.to(config.DEVICE)
    return model.to(config.DEVICE)

# Model Creation
def build_model_det() -> torch.nn.Module:
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
    num_targets = 2 * (sequence_length - past_trajectory)
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

model = build_model()
det_model = build_model_det().to(config.DEVICE)
# print(model)

if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    det_model = torch.nn.DataParallel(det_model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[1]).cuda()
    det_model = torch.nn.DataParallel(det_model, device_ids=[1]).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    # det_model = torch.nn.DataParallel(det_model).cuda()

model.to(config.DEVICE)
det_model.to(config.DEVICE)

# load the det_model checkpoint
print('Loading checkpoint')
det_checkpoint = torch.load(f"{config.DATASET_PATH}/{config.BEST_EPOCH}")
det_model_epoch = det_checkpoint['epoch']
# load model weights state_dict
det_model.load_state_dict(det_checkpoint['model_state_dict'])#, strict=False) # Error(s) in loading state_dict for ResNet: -> added strict-False
print('Loaded det_checkpoint at epoch', det_model_epoch)

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
    f.write("epoch, train_loss, validation_loss, learning_rate, best_epoch, best_val_loss, train_mse_loss, train_kl_loss, val_kl_loss, train_ade, train_fde, val_ade, val_fde\n")
    if config.LOSS_FUNCTION == "ADAPTIVE":
        params = list(parameters) + list(adaptive.parameters())

if config.LOSS_FUNCTION == "ADAPTIVE":
    optimizer = torch.optim.Adam(params, lr=config.LR, weight_decay=1e-3)


for (idx_1, layer_1), (det_idx_1, det_layer_1) in zip(
        enumerate(model.children()),
        enumerate(det_model.children())):
    MOPED_layer(layer_1, det_layer_1, config.delta)
    for (idx_2, layer_2), (det_idx_2, det_layer_2) in zip(
            enumerate(layer_1.children()),
            enumerate(det_layer_1.children())):
        MOPED_layer(layer_2, det_layer_2, config.delta)
        for (idx_3, layer_3), (det_idx_3, det_layer_3) in zip(
                enumerate(layer_2.children()),
                enumerate(det_layer_2.children())):
            MOPED_layer(layer_3, det_layer_3, config.delta)
            for (idx_4, layer_4), (det_idx_4, det_layer_4) in zip(
                    enumerate(layer_3.children()),
                    enumerate(det_layer_3.children())):
                MOPED_layer(layer_4, det_layer_4, config.delta)
                for (idx_5,
                        layer_5), (det_idx_5, det_layer_5) in zip(
                            enumerate(layer_4.children()),
                            enumerate(det_layer_4.children())):
                    MOPED_layer(layer_5, det_layer_5, config.delta)
                    for (idx_6,
                            layer_6), (det_idx_6, det_layer_6) in zip(
                                enumerate(layer_5.children()),
                                enumerate(det_layer_5.children())):
                        MOPED_layer(layer_6, det_layer_6,
                                    config.delta)

model.state_dict()
del det_model

logger_dir = os.path.join(config.OUTPUT_PATH, 'tb_logger')
if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)
tb_writer = SummaryWriter(logger_dir)


# training function
def fit(model, dataloader, data, epoch):
    print('Training')
    model.train()
    train_running_loss = 0.0
    mse_loss_total = 0.0
    kl_divergence_total = 0.0
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

        # image = image.to(f'cuda:{model.device_ids[0]}')
        outputs, kl = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # print(kl.size())
        if config.MULTIPLE_GPU:
            # scaled_kl = (kl.data[0].item() / config.BATCH_SIZE)
            scaled_kl = (kl.data.mean().item() / config.BATCH_SIZE)

        else:
            scaled_kl = (kl.data.item() / config.BATCH_SIZE)

        outputs = outputs.view(keypoints.size(0), -1)
        availability = availability.view(availability.size(0), -1)

        if config.LOSS_FUNCTION == "ADAPTIVE":
            loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
        elif config.LOSS_FUNCTION == "MSE":
            loss = criterion(outputs, keypoints)

        if config.train_evaluate:
            ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

        loss = loss * availability        
        mse_loss = loss.mean()
        scaled_kl = config.factor_kl*scaled_kl
        loss = loss + scaled_kl
        loss = loss.mean()
        train_running_loss += loss.item()
        mse_loss_total += mse_loss.item()
        kl_divergence_total += scaled_kl
        total_ade += ade.item()
        total_fde += fde.item()

        loss.backward()
        optimizer.step()

        # if tb_writer is not None:
        #     tb_writer.add_scalar('train/mse', mse_loss.item(), epoch)
        #     tb_writer.add_scalar('train/kl_div', scaled_kl, epoch)
        #     tb_writer.add_scalar('train/loss', loss.item(), epoch)
        #     tb_writer.flush()

    train_loss = train_running_loss/counter
    train_loss_mse = mse_loss_total/counter
    train_loss_kl = kl_divergence_total/counter
    train_ade = total_ade/counter
    train_fde = total_fde/counter

    return train_loss, train_loss_mse, train_loss_kl, train_ade, train_fde

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    mse_loss_total_val = 0.0
    kl_divergence_total_val = 0.0
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
            outputs, kl = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping

            if config.MULTIPLE_GPU:
                # scaled_kl = (kl.data[0].item() / config.BATCH_SIZE)
                scaled_kl = (kl.data.mean().item() / config.BATCH_SIZE)

            else:
                scaled_kl = (kl.data.item() / config.BATCH_SIZE)

            outputs = outputs.view(keypoints.size(0), -1)
            availability = availability.view(availability.size(0), -1)

            if config.LOSS_FUNCTION == "ADAPTIVE":
                loss = torch.mean(adaptive.lossfun((outputs - keypoints))) #[:,None] # (y_i - y)[:, None] # numpy array or tensor
            elif config.LOSS_FUNCTION == "MSE":
                loss = criterion(outputs, keypoints)

            if config.train_evaluate:
                ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())
            loss = loss * availability
            mse_loss_val = loss.mean()
            scaled_kl = config.factor_kl*scaled_kl
            loss = loss + scaled_kl
            loss = loss.mean()
            valid_running_loss += loss.item()
            mse_loss_total_val += mse_loss_val.item()
            kl_divergence_total_val += scaled_kl
            total_ade_ += ade.item()
            total_fde_ += fde.item()

            # if tb_writer is not None:
            #     tb_writer.add_scalar('val/mse', mse_loss_val.item(), epoch)
            #     tb_writer.add_scalar('val/kl_div', scaled_kl, epoch)
            #     tb_writer.add_scalar('val/loss', loss.item(), epoch)
            #     tb_writer.flush()

    valid_loss = valid_running_loss/counter
    val_loss_mse = mse_loss_total_val/counter
    val_loss_kl = kl_divergence_total_val/counter
    val_ade = total_ade_/counter
    val_fde = total_fde_/counter

    return valid_loss, val_loss_mse, val_loss_kl, val_ade, val_fde

train_loss = []
val_loss = []

prev_best_val_loss = 10000

for epoch in range(begin_epoch, config.EPOCHS):
    print(f"Epoch {epoch} of {config.EPOCHS}")

    if config.dynamic_lr:
        adjust_learning_rate(optimizer, epoch)

    train_epoch_loss, train_mse_loss, train_kl_loss, train_ade, train_fde = fit(model, train_loader, train_data, epoch)
    val_epoch_loss, val_mse_loss, val_kl_loss, val_ade, val_fde = validate(model, valid_loader, valid_data, epoch)

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


    f.write(str(epoch)+","+str(train_epoch_loss)+","+str(val_epoch_loss)+","+str(config.LR)+","+str(best_epoch_no)+","+str(best_val_loss)+","+str(train_mse_loss)+","+str(train_kl_loss)+","+str(val_mse_loss)+","+str(val_kl_loss)+","+str(train_ade)+","+str(train_fde)+","+str(val_ade)+","+str(val_fde)+"\n")
    print(f"Train Loss: {train_epoch_loss:.8f}")
    print(f'Val Loss: {val_epoch_loss:.8f}')
    print(f"MSE_Train Loss: {train_mse_loss:.8f}")
    print(f'MSE_Val Loss: {val_mse_loss:.8f}')
    print(f"KL_Train Loss: {train_kl_loss:.8f}")
    print(f'KL_Val Loss: {val_kl_loss:.8f}')
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
