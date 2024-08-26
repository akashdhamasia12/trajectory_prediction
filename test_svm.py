from typing_extensions import final
import config
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision.models.resnet import resnet50
from resnet50_dropout import resnet50
import torch.nn as nn
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if config.DATASET == "SHIFT":
    from dataset_shift import test_loader_out, test_loader_in
else:
    from dataset import test_loader, test_loader_noise, test_loader_training

import torchvision.models as models

import matplotlib.pyplot as plt
import matplotlib.image as Image
# from PIL import Image
import cv2
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
history_frames = past_trajectory*2 + 3
total_maneuvers = ["none", "straight", "right", "left"]

#set random seeds
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


def calculate_ade(outputs, targets):

    displacement = np.linalg.norm(outputs - targets, axis=1) 
    ade = np.mean(displacement)   
    return ade

def calculate_fde(outputs, targets):

    fde = np.linalg.norm(outputs[outputs.shape[0]-1] - targets[outputs.shape[0]-1])
    return fde

# torch.manual_seed(1)

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

if config.GPU:
    if config.MULTIPLE_GPU:
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=[config.cuda_device]).cuda()

model.to(config.DEVICE)

# # load the model checkpoint
print('Loading checkpoint')
checkpoint = torch.load(f"{config.OUTPUT_PATH}/{config.BEST_EPOCH}")
# checkpoint = torch.load(config.model_path)
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs_uncertainty_all/best_epoch_MSE_.pth")
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs/best_epoch_MSE_.pth")

model_epoch = checkpoint['epoch']
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])#, strict=False) # Error(s) in loading state_dict for ResNet: -> added strict-False
print('Loaded checkpoint at epoch', model_epoch)

print('run')
model.eval()

# validatioon function
def evaluate(model, dataloader, output_list, ade_list):

    with torch.no_grad():

        for i, data in enumerate(dataloader):

            image, keypoints, availability, seq_id, image_agent, centroid_current, history_traj, history_traj_availability, maneuver_label = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['seq_id'].to(config.DEVICE)), torch.squeeze(data['current_agent_i'].to(config.DEVICE)), torch.squeeze(data['centroid_current'].to(config.DEVICE)), torch.squeeze(data['history_traj'].to(config.DEVICE)), torch.squeeze(data['history_traj_availability'].to(config.DEVICE)), torch.squeeze(data['maneuver_label'].to(config.DEVICE))

            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            availability = availability.view(availability.size(0), -1)
            history_traj = history_traj.view(history_traj.size(0), -1)
            history_traj_availability = history_traj_availability.view(history_traj_availability.size(0), -1)

            outputs, outputs_ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            outputs = outputs.view(keypoints.size(0), -1)
            outputs_ = torch.squeeze(outputs_)

            image = image.detach().cpu().numpy()
            image_agent = image_agent.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            outputs_ = outputs_.detach().cpu().numpy()
            keypoints = keypoints.detach().cpu().numpy()
            availability = availability.detach().cpu().numpy()
            centroid_current = centroid_current.detach().cpu().numpy()
            history_traj = history_traj.detach().cpu().numpy()
            history_traj_availability = history_traj_availability.detach().cpu().numpy()
            maneuver_label = maneuver_label.detach().cpu().numpy()

            outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
            keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)

            outputs = outputs[np.where(availability == 1)]
            keypoints = keypoints[np.where(availability == 1)]
            history_traj = history_traj[np.where(history_traj_availability == 1)]

            output_list.append(outputs)

            outputs = outputs.reshape(-1,2)
            keypoints = keypoints.reshape(-1,2)
            history_traj = history_traj.reshape(-1,2)

            ade = calculate_ade(outputs, keypoints)
            # fde = calculate_fde(outputs, keypoints)

            ade_list.append(ade)

            # if i==100:
            #     return output_list, ade_list

        return output_list, ade_list


f=open(f"{config.plots}/ade-fde.txt","w+")
f.write("ade,svm_ade\n")

output_list = []
ade_list = []
test_loader_iter = test_loader_training

output_list, ade_list = evaluate(model, test_loader_iter, output_list, ade_list)

output_list = np.array(output_list)
ade_list = np.array(ade_list)

print("SVM training")
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(output_list, ade_list)

output_list = []
ade_list = []
test_loader_iter = test_loader

output_list, ade_list = evaluate(model, test_loader_iter, output_list, ade_list)
output_list = np.array(output_list)
ade_list = np.array(ade_list)

#prediction on validation dataset.
print("SVM prediction")
y_pred = regressor.predict(output_list)

for k in range(0, len(y_pred)):
    f.write(str(ade_list[k]) + "," + str(y_pred[k]) + "\n")


f.close()

# if config.DATASET == "SHIFT":
#     test_loader_iter = test_loader_out
# else:
#     test_loader_iter = test_loader_noise

# noise = True
# counter_1 = 0
# # validatioon function
# ade_list, fde_list, maneuver_gt, maneuver_outputs, variance_list, f = evaluate(model, test_loader_iter, ade_list, fde_list, maneuver_gt, maneuver_outputs, variance_list, f, counter_1, noise)

# cm = confusion_matrix(maneuver_gt, maneuver_outputs)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# # print(cm.diagonal())

# f.write("total_ADE =" + str(sum(ade_list)/len(ade_list)) +"\n")
# f.write("total_FDE =" + str(sum(fde_list)/len(fde_list)) +"\n")
# f.write("total_uncertainty =" + str(sum(variance_list)/len(variance_list)) +"\n")

# if config.MANEUVER_PRESENT:
#     f.write("Straight_turn_accuracy =" + str(cm.diagonal()[0]) +"\n")
#     f.write("Right_turn_accuracy =" + str(cm.diagonal()[1]) +"\n")
#     f.write("Left_turn_accuracy =" + str(cm.diagonal()[2]) +"\n")
#     f.write("Maneuver_classification =" + str(accuracy_score(maneuver_gt, maneuver_outputs)) +"\n")
# f.close()

# print("total_ADE", sum(ade_list)/len(ade_list))
# print("total_FDE", sum(fde_list)/len(fde_list))
# print("total_uncertainty", sum(variance_list)/len(variance_list))

# if config.MANEUVER_PRESENT:
#     print("Straight_turn_accuracy", cm.diagonal()[0])
#     print("Right_turn_accuracy", cm.diagonal()[1])
#     print("Left_turn_accuracy", cm.diagonal()[2])
#     print("Maneuver_classification", accuracy_score(maneuver_gt, maneuver_outputs))


