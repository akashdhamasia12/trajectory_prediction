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

import config_tuning
from torch.utils.data import Dataset, DataLoader
from dataset import FaceKeypointDataset
# from dataset import train_data, train_loader, valid_data, valid_loader
from os import path

from tqdm import tqdm
import utils
import robust_loss_pytorch
import math

from torch.utils.tensorboard import SummaryWriter

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


sequence_length = config_tuning.SEQ_LENGTH
past_trajectory = config_tuning.HISTORY

#set random seeds
random.seed(config_tuning.seed)
np.random.seed(config_tuning.seed)
torch.manual_seed(config_tuning.seed)
torch.cuda.manual_seed(config_tuning.seed)
torch.cuda.manual_seed_all(config_tuning.seed)
print("Random seed", config_tuning.seed)

learning_rate = config_tuning.LR


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
    outputs_ = ((outputs + 1)/2)*int(config_tuning.IMAGE_SIZE)
    keypoints_ = ((keypoints + 1)/2)*int(config_tuning.IMAGE_SIZE)

    availability_ = availability.bool()
    outputs_ = outputs_[availability_].view(outputs.size(0), -1)
    keypoints_ = keypoints_[availability_].view(outputs.size(0), -1)

    ade = calculate_ade(outputs_, keypoints_)
    fde = calculate_fde(outputs_, keypoints_)

    return ade, fde

def train_test_split(sequences, split):

    len_data = len(sequences)

    # calculate the validation data sample length
    valid_split = int(len_data * split)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    training_samples = sequences[:train_split]
    valid_samples = sequences[-valid_split:]
    return training_samples, valid_samples

#new (specific to lyft, data augmentation cannot be applied as we donot have the masks images of the future trajectories)
def create_sequences(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    all_seq = []
    counter_i = 0

    for i in range(0, data.shape[1], sequence_length):
        sequence = []
        if (i + sequence_length) <= data.shape[1]:
            for j in range(sequence_length):
                if config_tuning.MANEUVER_PRESENT:
                    sequence.append([data[0][i+j], data[1][i+j], data[2][i+j], data[3][i+j], data[4][i+j], data[5][i+j]]) #frameid,x,y,avail,seqid,turn
                else:
                    sequence.append([data[0][i+j], data[1][i+j], data[2][i+j], data[3][i+j], data[4][i+j]]) #frameid,x,y,avail,seqid,turn
            all_seq.append([counter_i, sequence])
            counter_i=counter_i+1
        else:
            break
    print("number of total sequences", len(all_seq))

    shuffle_file = config_tuning.DATASET_PATH + "/shuffle_file.csv"

    if path.isfile(shuffle_file):
        random_agents = np.genfromtxt(shuffle_file, delimiter=',')
        selected_elements = [all_seq[int(index)] for index in random_agents]
    else:
        random_agents = np.random.choice(len(all_seq), size=len(all_seq), replace=False)
        selected_elements = [all_seq[index] for index in random_agents]
        np.savetxt(shuffle_file, random_agents, delimiter=",")

    return selected_elements


def load_data():

    total_seq = create_sequences(f"{config_tuning.DATASET_PATH}/train_csvs/train.csv")

    num_of_sequences = int(len(total_seq) * config_tuning.num_sequences / 100)
    print("num of sequences selected ", num_of_sequences)

    training_samples, valid_samples = train_test_split(total_seq[:num_of_sequences], config_tuning.TEST_SPLIT)

    # initialize the dataset - `FaceKeypointDataset()`
    train_data = FaceKeypointDataset(training_samples, 
                                    f"{config_tuning.DATASET_PATH}")
    valid_data = FaceKeypointDataset(valid_samples, 
                                    f"{config_tuning.DATASET_PATH}")

    # test_data = FaceKeypointDataset(total_seq, 
    #                                  f"{config.DATASET_PATH}")

    return train_data, valid_data

# Model Creation
def build_model(prob) -> torch.nn.Module:
    # load pre-trained Conv2D model
    # '''
    model = resnet50(pretrained=True, p=prob)

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

# training function
def fit(model, dataloader, data, epoch, optimizer, criterion):
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
        image, keypoints, availability = data['image'].to(config_tuning.DEVICE), torch.squeeze(data['keypoints'].to(config_tuning.DEVICE)), torch.squeeze(data['availability'].to(config_tuning.DEVICE))
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        outputs = outputs.view(keypoints.size(0), -1)
        availability = availability.view(availability.size(0), -1)

        loss = criterion(outputs, keypoints)

        if config_tuning.train_evaluate:
            ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

        loss = loss * availability        
        loss = loss.mean() 
        train_running_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss/counter
    train_ade = total_ade/counter
    train_fde = total_fde/counter

    return train_loss, train_ade, train_fde

# validatioon function
def validate(model, dataloader, data, epoch, optimizer, criterion):
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
            image, keypoints, availability = data['image'].to(config_tuning.DEVICE), torch.squeeze(data['keypoints'].to(config_tuning.DEVICE)), torch.squeeze(data['availability'].to(config_tuning.DEVICE))
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            outputs = outputs.view(keypoints.size(0), -1)
            availability = availability.view(availability.size(0), -1)

            loss = criterion(outputs, keypoints)

            if config_tuning.train_evaluate:
                ade, fde = evaluate(outputs.clone(), keypoints.clone(), availability.clone())

            loss = loss * availability
            loss = loss.mean()
            valid_running_loss += loss.item()
            total_ade_ += ade.item()
            total_fde_ += fde.item()

    valid_loss = valid_running_loss/counter
    val_ade = total_ade_/counter
    val_fde = total_fde_/counter

    return valid_loss, val_ade, val_fde

def train_traj(config, checkpoint_dir=None):

    model = build_model(config["prob"]).to(config_tuning.DEVICE)
    # print(model)
    if config_tuning.MULTIPLE_GPU:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=[config_tuning.cuda_device]).cuda()
        # model = torch.nn.DataParallel(model).cuda()
    model.to(config_tuning.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss(reduction="none") # criterion = nn.SmoothL1Loss()

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)


    train_data, valid_data = load_data()
    # prepare data loaders
    train_loader = DataLoader(train_data, 
                            batch_size=int(config["batch_size"]), 
                            shuffle=True)

    valid_loader = DataLoader(valid_data, 
                            batch_size=int(config["batch_size"]), 
                            shuffle=False)


    for epoch in range(0, config_tuning.EPOCHS):
        print(f"Epoch {epoch} of {config_tuning.EPOCHS}")

        train_epoch_loss, train_ade, train_fde = fit(model, train_loader, train_data, epoch, optimizer, criterion)
        val_epoch_loss, val_ade, val_fde = validate(model, valid_loader, valid_data, epoch, optimizer, criterion)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(math.exp(val_epoch_loss)), accuracy=val_ade)
    print("Finished Training")

def main(num_samples=10, max_num_epochs=20, gpus_per_trial=1):
    config = {
        "prob": tune.loguniform(0.1, 0.3),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        tune.with_parameters(train_traj),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=20, gpus_per_trial=1)