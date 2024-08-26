import config
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
import torch.nn as nn
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataset import test_loader
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

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[1][0]-lineA[0][0]), (lineA[1][1]-lineA[0][1])]
    vB = [(lineB[1][0]-lineB[0][0]), (lineB[1][1]-lineB[0][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees

    # print(cos_, magA, magB)

    if -1<=cos_<=1: 
        angle = math.acos(cos_) #dot_prod/magB/magA)
        # angle = math.acos(dot_prod/magB/magA) #why if gives error, when cos_ = -1, seq=9957612?
    else:
        return 0, 0
    # # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    angle_tan = math.atan2(vB[1], vB[0]) - math.atan2(vA[1], vA[0])

    # if ang_deg-180>=0:
    #     # As in if statement
    #     return 360 - ang_deg
    # else: 
    #     return ang_deg

    return ang_deg, angle_tan

def cal_maneuver(angle_btw_lines, angle_btw_lines_tan):

    if angle_btw_lines < 5: #almost straight
        turn = 1 #straight
    elif angle_btw_lines_tan > 0 and angle_btw_lines > 5: #Right Turn
        turn = 2 #right
    elif angle_btw_lines_tan < 0 and angle_btw_lines > 5: #Left Turn
        turn = 3 #left

    return turn

def trajectories_plot(image, outputs_straight, outputs_right, outputs_left, targets, j, image_agent_all, centroid_, ade_straight, fde_straight, ade_right, fde_right, ade_left, fde_left, history_traj, maneuver_label, maneuver_pred):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints for each test image in the batch.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Seq {}".format(j))

    img_map = np.copy(image[0, history_frames-3:history_frames, :, :])
    img_map = np.transpose(img_map, (1, 2, 0))
    img_map1 = np.copy(img_map)

    img_agent = np.copy(image[0, 0, :, :])
    img_agent = img_agent > 0.2
    img_map[img_agent] = [0.0, 0.0, 1.0] #(RGB) #blue for agents

    image_agent_all = image_agent_all > 40
    img_map1[image_agent_all] = [0.0, 0.0, 1.0] #(RGB) #blue for agents

    img_ego = np.copy(image[0, past_trajectory, :, :])
    img_ego = img_ego > 0.2
    img_map[img_ego] = [0.0, 1.0, 0.0] #(RGB) #green for ego
    img_map1[img_ego] = [0.0, 1.0, 0.0] #(RGB) #green for ego

    if config.NEIGHBOUR_RADIUS > 0:
        circle2 = plt.Circle((int(centroid_[0]), int(centroid_[1])), int(config.NEIGHBOUR_RADIUS), color='b', fill=False)
        ax2.add_patch(circle2)

    ax1.imshow(img_map)
    ax2.imshow(img_map1)
    
    # outputs = outputs.reshape(-1,2)
    # targets = targets.reshape(-1,2)
    # history_traj = history_traj.reshape(-1,2)

    ax3.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))

    t_= "ADE = " + str(round(ade_straight,3)) + "\nFDE = " + str(round(fde_straight,3)) + "\nManeuver_gt = " + total_maneuvers[maneuver_label] + "\nManeuver_pred = " + total_maneuvers[maneuver_pred] 
    ax3.text(10,10,t_)

    ax1.plot(targets[:,0], targets[:,1], color='green', marker='o', linewidth=0.1, markersize=1, label='Ground-Truth')
    ax1.plot(outputs_straight[:,0], outputs_straight[:,1], color='red', marker='x', linewidth=0.1, markersize=1.4, label='Predictions')
    ax1.plot(outputs_right[:,0], outputs_right[:,1], color='blue', marker='x', linewidth=0.1, markersize=1.4, label='Predictions')
    ax1.plot(outputs_left[:,0], outputs_left[:,1], color='purple', marker='x', linewidth=0.1, markersize=1.4, label='Predictions')
    ax1.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', linewidth=0.1, markersize=1, label='history')

    ax2.plot(targets[:,0], targets[:,1], color='green', marker='o', linewidth=0.1, markersize=1, label='Ground-Truth')
    ax2.plot(outputs_straight[:,0], outputs_straight[:,1], color='red', marker='x', linewidth=0.1, markersize=1.4, label='Predictions')
    ax2.plot(outputs_right[:,0], outputs_right[:,1], color='blue', marker='x', linewidth=0.1, markersize=1.4, label='Predictions')
    ax2.plot(outputs_left[:,0], outputs_left[:,1], color='purple', marker='x', linewidth=0.1, markersize=1.4, label='Predictions')
    ax2.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', linewidth=0.1, markersize=1, label='history')

    ax3.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.5, linewidth=0.8, label='Ground-Truth')
    ax3.plot(outputs_straight[:,0], outputs_straight[:,1], color='red', marker='x', markersize=0.8, linewidth=0.8, label='Predictions')
    ax3.plot(outputs_right[:,0], outputs_right[:,1], color='blue', marker='x', markersize=0.5, linewidth=0.8, label='Predictions')
    ax3.plot(outputs_left[:,0], outputs_left[:,1], color='purple', marker='x', markersize=0.5, linewidth=0.8, label='Predictions')
    ax3.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', markersize=0.5, linewidth=0.8, label='history')

    # for i in range(outputs.shape[0]):
    #     ax3.annotate(str(i), # this is the text
    #                 (outputs[i][0],outputs[i][1]), # this is the point to label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(0,10), # distance from text to points (x,y)
    #                 ha='center') # horizontal alignment can be left, right or center

    #     ax3.annotate(str(i), # this is the text
    #                 (targets[i][0],targets[i][1]), # this is the point to label
    #                 textcoords="offset points", # how to position the text
    #                 xytext=(0,10), # distance from text to points (x,y)
    #                 ha='center') # horizontal alignment can be left, right or center


    plt.savefig(f"{config.OUTPUT_PATH}/plots/seq_" + str(j) + ".png")
    plt.close()

'''
im = mimage.imread(datafile)
fig, ax = plt.subplots(figsize=(10,10))
myaximage = ax.imshow(im,
aspect='auto',
extent=(-1, 1, -1, 1))
'''

def calculate_ade(outputs, targets):

    # batch_size = targets.size(0)
    # targets = targets.reshape(-1,2)
    # outputs = outputs.reshape(-1,2)
    # displacement = torch.norm(outputs - targets, dim=1)
    displacement = np.linalg.norm(outputs - targets, axis=1) 
    ade = np.mean(displacement)   

    return ade
    # return ade.item()

def calculate_fde(outputs, targets):

    # batch_size = targets.size(0)
    # targets = targets.reshape(-1,2)
    # outputs = outputs.reshape(-1,2)
    # fde = torch.norm(outputs[outputs.size(0)-1] - targets[outputs.size(0)-1])
    fde = np.linalg.norm(outputs[outputs.shape[0]-1] - targets[outputs.shape[0]-1])

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
    num_targets = 2 * (sequence_length - past_trajectory) * config.multimodal
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    # change input channels number to match the rasterizer's output
    # model = models.densenet161(pretrained=True)

    # num_in_channels = 3 + (2*past_trajectory)
    # model.features.conv0 = nn.Conv2d(
    #     num_in_channels,
    #     model.features.conv0.out_channels,
    #     kernel_size=model.features.conv0.kernel_size,
    #     stride=model.features.conv0.stride,
    #     padding=model.features.conv0.padding,
    #     bias=False,
    # )
    # # change output size to (X, Y) * number of future states
    # num_targets = 2 * (sequence_length - past_trajectory)
    # model.classifier = nn.Linear(in_features=2208, out_features=num_targets)

    return model

model = build_model().to(config.DEVICE)

if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # model = torch.nn.DataParallel(model).cuda()

model.to(config.DEVICE)
# print(model)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if
#                            p.requires_grad)
# print("Total number of trainable parameters: ", pytorch_total_params)

# # load the model checkpoint
print('Loading checkpoint')
checkpoint = torch.load(f"{config.OUTPUT_PATH}/{config.BEST_EPOCH}")
# checkpoint = torch.load(f"{config.DATASET_PATH}/{config.BEST_EPOCH}")
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/best_epoch_MSE_.pth")
model_epoch = checkpoint['epoch']
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])#, strict=False) # Error(s) in loading state_dict for ResNet: -> added strict-False
# model.load_state_dict(checkpoint['model_state_dict'], strict=False)

print('Loaded checkpoint at epoch', model_epoch)

print('run')

model.eval()
counter = 0
# num_batches = int(len(test_data)/test_loader.batch_size)

ade_list_straight = []
ade_list_right = []
ade_list_left = []

fde_list_straight = []
fde_list_right = []
fde_list_left = []

maneuver_gt = []
maneuver_outputs = []

f=open(f"{config.OUTPUT_PATH}/ade-fde.txt","w+")
f.write("seq_id,ade,fde,maneuver_gt,maneuver_pred\n")

with torch.no_grad():
    for i, data in enumerate(test_loader):

        image, keypoints, availability, seq_id, image_agent, centroid_current, history_traj, history_traj_availability, maneuver_label = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['seq_id'].to(config.DEVICE)), torch.squeeze(data['current_agent_i'].to(config.DEVICE)), torch.squeeze(data['centroid_current'].to(config.DEVICE)), torch.squeeze(data['history_traj'].to(config.DEVICE)), torch.squeeze(data['history_traj_availability'].to(config.DEVICE)), torch.squeeze(data['maneuver_label'].to(config.DEVICE))

        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        availability = availability.view(availability.size(0), -1)
        history_traj = history_traj.view(history_traj.size(0), -1)
        history_traj_availability = history_traj_availability.view(history_traj_availability.size(0), -1)

        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        outputs = outputs.view(keypoints.size(0), -1)

        image = image.detach().cpu().numpy()
        image_agent = image_agent.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        keypoints = keypoints.detach().cpu().numpy()
        availability = availability.detach().cpu().numpy()
        centroid_current = centroid_current.detach().cpu().numpy()
        history_traj = history_traj.detach().cpu().numpy()
        history_traj_availability = history_traj_availability.detach().cpu().numpy()
        maneuver_label = maneuver_label.detach().cpu().numpy()

        outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
        keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)

        outputs_straight = outputs[0:(sequence_length-past_trajectory)*2]
        outputs_right = outputs[(sequence_length-past_trajectory)*2:(sequence_length-past_trajectory)*2*(config.multimodal-1)]
        outputs_left = outputs[(sequence_length-past_trajectory)*2*(config.multimodal-1):]

        keypoints_straight = keypoints[0:(sequence_length-past_trajectory)*2]
        keypoints_right = keypoints[(sequence_length-past_trajectory)*2:(sequence_length-past_trajectory)*2*(config.multimodal-1)]
        keypoints_left = keypoints[(sequence_length-past_trajectory)*2*(config.multimodal-1):]

        availability_straight = availability[0:(sequence_length-past_trajectory)*2]
        availability_right = availability[(sequence_length-past_trajectory)*2:(sequence_length-past_trajectory)*2*(config.multimodal-1)]
        availability_left = availability[(sequence_length-past_trajectory)*2*(config.multimodal-1):]

        outputs_straight = outputs_straight[np.where(availability_straight == 1)]
        outputs_right = outputs_right[np.where(availability_right == 1)]
        outputs_left = outputs_left[np.where(availability_left == 1)]

        keypoints_straight = keypoints_straight[np.where(availability_straight == 1)]
        keypoints_right = keypoints_right[np.where(availability_right == 1)]
        keypoints_left = keypoints_left[np.where(availability_left == 1)]

        history_traj = history_traj[np.where(history_traj_availability == 1)]

        outputs_straight = outputs_straight.reshape(-1,2)
        outputs_right = outputs_right.reshape(-1,2)
        outputs_left = outputs_left.reshape(-1,2)

        keypoints_straight = keypoints_straight.reshape(-1,2)
        keypoints_right = keypoints_right.reshape(-1,2)
        keypoints_left = keypoints_left.reshape(-1,2)

        history_traj = history_traj.reshape(-1,2)

        ade_straight = calculate_ade(outputs_straight, keypoints_straight) * config.IMAGE_FACTOR
        fde_straight = calculate_fde(outputs_straight, keypoints_straight) * config.IMAGE_FACTOR

        ade_right = calculate_ade(outputs_right, keypoints_right) * config.IMAGE_FACTOR
        fde_right = calculate_fde(outputs_right, keypoints_right) * config.IMAGE_FACTOR

        ade_left = calculate_ade(outputs_left, keypoints_left) * config.IMAGE_FACTOR
        fde_left = calculate_fde(outputs_left, keypoints_left) * config.IMAGE_FACTOR

        ade_list_straight.append(ade_straight)
        fde_list_straight.append(fde_straight)

        ade_list_right.append(ade_right)
        fde_list_right.append(fde_right)

        ade_list_left.append(ade_left)
        fde_list_left.append(fde_left)

        if config.MANEUVER_PRESENT:
            line_history = [history_traj[past_trajectory-1], history_traj[0]]
            line_future = [outputs[0], outputs[-1]]
            angle_btw_lines, angle_btw_lines_tan = ang(line_history, line_future)
            maneuver_pred = cal_maneuver(angle_btw_lines, angle_btw_lines_tan)
            maneuver_gt.append(maneuver_label)
            maneuver_outputs.append(maneuver_pred)
        else:
            maneuver_label = 0
            maneuver_pred = 0

        print("seq_id, ADE_straight, FDE_straight, ADE_right, FDE_right, ADE_left, FDE_left, maneuver_gt, maneuver_pred = ", seq_id.item(), ade_straight, fde_straight, ade_right, fde_right, ade_left, fde_left, maneuver_label, maneuver_pred)
        f.write(str(seq_id.item())+","+str(ade_straight)+","+str(fde_straight)+","+ str(ade_right)+","+str(fde_right)+","+ str(ade_left)+","+str(fde_left)+","+ str(maneuver_label)+","+str(maneuver_pred)+"\n")

        trajectories_plot(image, outputs_straight, outputs_right, outputs_left, keypoints_straight, seq_id.item(), image_agent, centroid_current, ade_straight, fde_straight, ade_right, fde_right, ade_left, fde_left, history_traj, maneuver_label, maneuver_pred)



cm = confusion_matrix(maneuver_gt, maneuver_outputs)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm.diagonal())

f.write("total_ADE_straight =" + str(sum(ade_list_straight)/len(ade_list_straight)) +"\n")
f.write("total_FDE_straight =" + str(sum(fde_list_straight)/len(fde_list_straight)) +"\n")
f.write("total_ADE_right =" + str(sum(ade_list_right)/len(ade_list_right)) +"\n")
f.write("total_FDE_right =" + str(sum(fde_list_right)/len(fde_list_right)) +"\n")
f.write("total_ADE_left =" + str(sum(ade_list_left)/len(ade_list_left)) +"\n")
f.write("total_FDE_left =" + str(sum(fde_list_left)/len(fde_list_left)) +"\n")

if config.MANEUVER_PRESENT:
    f.write("Straight_turn_accuracy =" + str(cm.diagonal()[0]) +"\n")
    f.write("Right_turn_accuracy =" + str(cm.diagonal()[1]) +"\n")
    f.write("Left_turn_accuracy =" + str(cm.diagonal()[2]) +"\n")
    f.write("Maneuver_classification =" + str(accuracy_score(maneuver_gt, maneuver_outputs)) +"\n")
f.close()

# print("total_ADE", sum(ade_list)/len(ade_list))
# print("total_FDE", sum(fde_list)/len(fde_list))

if config.MANEUVER_PRESENT:
    print("Straight_turn_accuracy", cm.diagonal()[0])
    print("Right_turn_accuracy", cm.diagonal()[1])
    print("Left_turn_accuracy", cm.diagonal()[2])
    print("Maneuver_classification", accuracy_score(maneuver_gt, maneuver_outputs))


