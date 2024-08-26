#from train import model, val_loss
import config
import torch
# from dataset import create_sequences, FaceKeypointDataset 
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
import torch.nn as nn
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from dataset import train_data, train_loader, valid_data, valid_loader, test_loader
from dataset import test_loader

import matplotlib.pyplot as plt
import matplotlib.image as Image
# from PIL import Image
# import cv2

# from torch.utils.data import DataLoader
# import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os

# set env variable for data
# os.environ["L5KIT_DATA_FOLDER"] = "/home/adhamasia/Datasets/lyft/lyft_prediction/"
# os.environ["L5KIT_DATA_FOLDER"] = "/mnt/raid0/trajectorypred/Datasets/Lyft/lyft_prediction/"
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/raid0/trajectorypred/Datasets/Lyft/lyft_prediction_new/"

# get config
cfg = load_config_data("./visualisation_config.yaml")
# cfg = load_config_data("./agent_motion_config.yaml")

print(cfg)

print(f'current raster_param:\n')
for k,v in cfg["raster_params"].items():
    print(f"{k}:{v}")

dm = LocalDataManager()
dataset_path = dm.require(cfg["train_data_loader"]["key"])
# dataset_path = dm.require(cfg["test_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

rasterizer = build_rasterizer(cfg, dm)
agent_train_dataset = AgentDataset(cfg, zarr_dataset, rasterizer)


sequence_length = 41
past_trajectory = 11

def trajectories_plot(image, outputs, targets, j):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints for each test image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    # image = image.detach().cpu()
    # outputs = outputs.detach().cpu().numpy()
    # targets = targets.detach().cpu().numpy()
    # just get a single datapoint from each batch
    # img = image[0, 20:23, :, :]
    # output_keypoint = outputs[0]
    # orig_keypoint = orig_keypoints[0]
    # img = np.array(img, dtype='float32')
    # img = np.transpose(img, (1, 2, 0))
    # fig, ax = plt.subplots(figsize=(10,10)) #new
    # #plt.imshow(img) #original
    # myaximage = ax.imshow(img, aspect='auto', extent=(-1, 1, -1, 1)) #new

    # output_keypoint = output_keypoint.reshape(-1, 2)
    # orig_keypoint = orig_keypoint.reshape(-1, 2)
    outputs = outputs.reshape(-1,2)
    targets = targets.reshape(-1,2)
    plt.plot(outputs[:,0], outputs[:,1], color='green', marker='o', label='Ground-Truth')
    plt.plot(targets[:,0], targets[:,1], color='red', marker='x', label='Predictions')

    for i in range(outputs.shape[0]):
        plt.annotate(str(i), # this is the text
                    (outputs[i][0],outputs[i][1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

        plt.annotate(str(i), # this is the text
                    (targets[i][0],targets[i][1]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.savefig(f"{config.OUTPUT_PATH}/plots/seq_" + str(j) + ".png")
    plt.close()

    # for p in range(output_keypoint.shape[0]):
    #     plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
    #     plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
    # plt.savefig(f"{config.OUTPUT_PATH}/test.png")
    # plt.close()

'''
im = mimage.imread(datafile)
fig, ax = plt.subplots(figsize=(10,10))
myaximage = ax.imshow(im,
aspect='auto',
extent=(-1, 1, -1, 1))
'''

def calculate_ade(outputs, targets):

    # batch_size = targets.size(0)
    targets = targets.reshape(-1,2)
    outputs = outputs.reshape(-1,2)
    # displacement = torch.norm(outputs - targets, dim=1)
    displacement = np.linalg.norm(outputs - targets, axis=1) 

    # pred_length = sequence_length - past_trajectory
    # error = torch.zeros(pred_length)

    # for tstep in range(pred_length):

    #     pred_pos = outputs[tstep, :]
    #     true_pos = targets[tstep, :]
    #     error[tstep] = torch.norm(pred_pos - true_pos, p=2)

    # ade = torch.mean(displacement)   
    ade = np.mean(displacement)   

    return ade
    # return ade.item()

def calculate_fde(outputs, targets):

    # batch_size = targets.size(0)
    targets = targets.reshape(-1,2)
    outputs = outputs.reshape(-1,2)
    # fde = torch.norm(outputs[outputs.size(0)-1] - targets[outputs.size(0)-1])
    fde = np.linalg.norm(outputs[outputs.shape[0]-1] - targets[outputs.shape[0]-1])

    # pred_length = sequence_length - past_trajectory
    # # Last time-step
    # tstep = pred_length - 1
        
    # pred_pos = outputs[tstep, :]
    # true_pos = targets[tstep, :]
    # fde = torch.norm(pred_pos - true_pos, p=2)

    # return fde.item()
    return fde

torch.manual_seed(1)

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
    num_targets = 2 * (sequence_length - past_trajectory)
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = build_model().to(device)
model = build_model().to(config.DEVICE)
model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

# print(model)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if
#                            p.requires_grad)
# print("Total number of trainable parameters: ", pytorch_total_params)

# Call for best model
#min_loss = min(val_loss)
#min_index = val_loss.index(min_loss)
min_index = 25 #150 #29
# load the model checkpoint
print('Loading checkpoint')
checkpoint = torch.load(f"{config.OUTPUT_PATH}/model_{min_index}.pth", map_location=torch.device('cpu'))
model_epoch = checkpoint['epoch']
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
print('Loaded checkpoint at epoch', model_epoch)

# test_samples = create_sequences(f"{config.ROOT_PATH}/lane_change_test.csv")
# # initialize the dataset - `FaceKeypointDataset()`
# test_data = FaceKeypointDataset(test_samples, 
#                                  f"{config.ROOT_PATH}")

# # prepare data loaders
# test_loader = DataLoader(test_data, 
#                           batch_size=1, 
#                           shuffle=False)

print('run')

model.eval()
counter = 0
# num_batches = int(len(test_data)/test_loader.batch_size)

ade_list = []
fde_list = []

f=open(f"{config.OUTPUT_PATH}/ade-fde.txt","w+")
f.write("seq_id,ade,fde\n")

with torch.no_grad():
    for i, data in enumerate(test_loader):

        image, keypoints, availability, seq_id = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['seq_id'].to(config.DEVICE))
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        availability = availability.view(availability.size(0), -1)

        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        outputs = outputs.view(keypoints.size(0), -1)

        outputs = outputs.detach().cpu().numpy()
        keypoints = keypoints.detach().cpu().numpy()
        availability = availability.detach().cpu().numpy()

        data = agent_train_dataset[seq_id.item()]
        im = data["image"].transpose(1, 2, 0)
        im = agent_train_dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
        # predicted = transform_points(outputs.reshape(-1,2) + data["centroid"][:2], data["world_to_image"])
        outputs = ((outputs + 1)/2)*255
        keypoints = ((keypoints + 1)/2)*255

        draw_trajectory(im, target_positions_pixels, data["target_yaws"], (0, 255, 0))
        draw_trajectory(im, outputs.reshape(-1,2), data["target_yaws"], (255, 0, 0))

        Image.imsave(f"{config.OUTPUT_PATH}/plots/seq_" + str(seq_id.item()) + "_map.png", im)

        outputs = outputs[np.where(availability == 1)]
        keypoints = keypoints[np.where(availability == 1)]

        # keypoints = keypoints * availability
        # outputs = outputs * availability

        ade = calculate_ade(outputs, keypoints)
        fde = calculate_fde(outputs, keypoints)
        print("seq_id, ADE, FDE = ", seq_id.item(), ade, fde)
        f.write(str(seq_id.item())+","+str(ade)+","+str(fde)+"\n")

        ade_list.append(ade)
        fde_list.append(fde)

        trajectories_plot(image, outputs, keypoints, seq_id.item())

f.close()
print("total_ADE", sum(ade_list)/len(ade_list))
print("total_FDE", sum(fde_list)/len(fde_list))


