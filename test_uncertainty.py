from typing_extensions import final
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

from bayesian_torch_.bayesian_torch.models.bayesian import resnet_variational_large as resnet
from bayesian_torch_.bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch_.bayesian_torch.layers import LinearReparameterization

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
history_frames = past_trajectory*2 + 3
total_maneuvers = ["none", "straight", "right", "left"]

prior_mu = 0.0
prior_sigma = 0.1
posterior_mu_init = 0.0
posterior_rho_init = -9.0

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

def trajectories_plot(image, outputs, targets, j, image_agent_all, centroid_, ade, fde, history_traj, maneuver_label, maneuver_pred, output_list, outputs_var_mean, final_index):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints for each test image in the batch.
    """

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig1, (ax3, ax4) = plt.subplots(1, 2)

    fig.suptitle("Seq {}".format(j))
    fig1.suptitle("Seq {}".format(j))

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
    ax4.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))

    # ax5.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))
    # ax6.set(xlim=(0, config.IMAGE_SIZE), ylim=(config.IMAGE_SIZE, 0))

    t_= "ADE = " + str(round(ade,3)) + "\nFDE = " + str(round(fde,3)) + "\nManeuver_gt = " + total_maneuvers[maneuver_label] + "\nManeuver_pred = " + total_maneuvers[maneuver_pred] 
    ax3.text(10,10,t_)

    ax1.plot(targets[:,0], targets[:,1], color='green', marker='o', linewidth=0.1, markersize=1, label='Ground-Truth')
    ax1.plot(outputs[:,0], outputs[:,1], color='red', marker='x', linewidth=0.1, markersize=1, label='Predictions')
    ax1.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', linewidth=0.1, markersize=1, label='history')

    ax2.plot(targets[:,0], targets[:,1], color='green', marker='o', linewidth=0.1, markersize=1, label='Ground-Truth')
    ax2.plot(outputs[:,0], outputs[:,1], color='red', marker='x', linewidth=0.1, markersize=1, label='Predictions')
    ax2.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', linewidth=0.1, markersize=1, label='history')

    ax3.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.3, linewidth=0.3, label='Ground-Truth')
    # ax3.plot(outputs[:,0], outputs[:,1], color='red', marker='x', markersize=1, label='Predictions')
    ax3.errorbar(outputs[:,0], outputs[:,1], outputs_var_mean, color='red', linestyle='None', marker='+', markersize=0.4, linewidth=0.4, label='Predictions') #capsize=0.2
    ax3.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', markersize=0.3, label='history')

    if final_index != -1:
        closest_output = output_list[final_index]
        closest_output = closest_output.reshape(-1,2)
        ax3.plot(closest_output[:,0], closest_output[:,1], color='b', marker='+', markersize=0.3, linewidth=0.5, label='Predictions')

    ax4.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.1, linewidth=0.5, label='Ground-Truth')
    ax4.plot(outputs[:,0], outputs[:,1], color='red', linestyle='None', marker='x', markersize=0.3, linewidth=0.5, label='Predictions') #capsize=0.2
    ax4.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='x', markersize=0.3, label='history')

    # ax5.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.3, linewidth=0.3, label='Ground-Truth')
    # # ax3.plot(outputs[:,0], outputs[:,1], color='red', marker='x', markersize=1, label='Predictions')
    # ax5.errorbar(outputs[:,0], outputs[:,1], outputs_var_mean, color='red', linestyle='None', marker='+', markersize=0.4, linewidth=0.4, label='Predictions') #capsize=0.2
    # ax5.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='+', markersize=0.3, label='history')

    # if final_index != -1:
    #     closest_output = output_list[final_index]
    #     closest_output = closest_output.reshape(-1,2)
    #     ax5.plot(closest_output[:,0], closest_output[:,1], color='b', marker='+', markersize=0.3, linewidth=0.5, label='Predictions')

    # ax6.plot(targets[:,0], targets[:,1], color='green', marker='o', markersize=0.1, linewidth=0.5, label='Ground-Truth')
    # ax6.plot(outputs[:,0], outputs[:,1], color='red', linestyle='None', marker='x', markersize=0.3, linewidth=0.5, label='Predictions') #capsize=0.2
    # ax6.plot(history_traj[:,0], history_traj[:,1], color='cyan', marker='x', markersize=0.3, label='history')

    for multiple_output in output_list:
        multiple_output = multiple_output.reshape(-1,2)
        c = np.random.rand(3,)
        ax1.plot(multiple_output[:,0], multiple_output[:,1], color=c, marker='x', markersize=0.2, linewidth=1, label='Predictions')
        ax4.plot(multiple_output[:,0], multiple_output[:,1], color=c, marker='x', markersize=0.2, linewidth=1, label='Predictions')
        # ax6.plot(multiple_output[:,0], multiple_output[:,1], color=c, marker='x', markersize=0.2, linewidth=1, label='Predictions')


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


    fig.savefig(f"{config.OUTPUT_PATH}/plots/seq_" + str(j) + ".png")
    fig1.savefig(f"{config.OUTPUT_PATH}/plots/seq_" + str(j) + "_1.png")
    # plt.close()
    plt.close('all')

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
def build_model():

    # # load pre-trained Conv2D model
    # model = resnet50(pretrained=True)

    # # change input channels number to match the rasterizer's output
    # num_in_channels = 3 + (2*past_trajectory)
    # model.conv1 = nn.Conv2d(
    #     num_in_channels,
    #     model.conv1.out_channels,
    #     kernel_size=model.conv1.kernel_size,
    #     stride=model.conv1.stride,
    #     padding=model.conv1.padding,
    #     bias=False,
    # )
    # # change output size to (X, Y) * number of future states
    # num_targets = 2 * (sequence_length - past_trajectory)
    # model.fc = nn.Linear(in_features=2048, out_features=num_targets)

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


    return model.to(config.DEVICE)

model = build_model()
# print(model)

if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

model.to(config.DEVICE)

# # load the model checkpoint
print('Loading checkpoint')
checkpoint = torch.load(f"{config.OUTPUT_PATH}/{config.BEST_EPOCH}")
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs_uncertainty_all/best_epoch_MSE_.pth")
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs/best_epoch_MSE_.pth")

model_epoch = checkpoint['epoch']
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])#, strict=False) # Error(s) in loading state_dict for ResNet: -> added strict-False
print('Loaded checkpoint at epoch', model_epoch)

print('run')

# model.eval()
counter = 0
# num_batches = int(len(test_data)/test_loader.batch_size)

ade_list = []
fde_list = []
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

        image_agent = image_agent.detach().cpu().numpy()
        availability = availability.detach().cpu().numpy()
        centroid_current = centroid_current.detach().cpu().numpy()
        history_traj = history_traj.detach().cpu().numpy()
        history_traj_availability = history_traj_availability.detach().cpu().numpy()
        maneuver_label = maneuver_label.detach().cpu().numpy()
        keypoints = keypoints.detach().cpu().numpy()
        keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)

        outputs_list = []
        ade_final = 1000
        fde_final = 1000

        for mc_run in range(config.num_monte_carlo):
            model.eval()
            outputs, _ = model.forward(image)
            outputs = outputs.view(keypoints.shape[0], -1)
            outputs = outputs.detach().cpu().numpy()
            outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
            outputs = outputs[np.where(availability == 1)]
            keypoints1 = keypoints[np.where(availability == 1)]
            outputs_list.append(outputs)
            ade_value = calculate_ade(outputs.reshape(-1,2), keypoints1.reshape(-1,2)) * config.IMAGE_FACTOR
            fde_value = calculate_fde(outputs.reshape(-1,2), keypoints1.reshape(-1,2)) * config.IMAGE_FACTOR

            if ade_value < ade_final:
                ade_final = ade_value
                ade_index = mc_run

            if fde_value < fde_final:
                fde_final = fde_value
                fde_index = mc_run

 
        image = image.detach().cpu().numpy()

        keypoints = keypoints[np.where(availability == 1)]
        history_traj = history_traj[np.where(history_traj_availability == 1)]

        keypoints = keypoints.reshape(-1,2)
        history_traj = history_traj.reshape(-1,2)

        outputs_mean = np.mean(np.asarray(outputs_list), axis=0)
        outputs_var = np.std(np.asarray(outputs_list), axis=0)

        # outputs_mean = np.asarray(outputs_list[0])
        outputs_mean = outputs_mean.reshape(-1,2)
        outputs_var = outputs_var.reshape(-1,2)
        outputs_var_mean = np.mean(outputs_var, axis=1)
        
        ade = calculate_ade(outputs_mean, keypoints) * config.IMAGE_FACTOR
        fde = calculate_fde(outputs_mean, keypoints) * config.IMAGE_FACTOR

        if ade < ade_final:
            ade_final = ade
            ade_index = -1

        if fde < fde_final:
            fde_final = fde
            fde_index = -1

        if ade_index == fde_index:
            final_index = ade_index
        else:
            final_index = ade_index

        ade_list.append(ade)
        fde_list.append(fde)

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

        print("seq_id, ADE, FDE, maneuver_gt, maneuver_pred = ", seq_id.item(), ade, fde, maneuver_label, maneuver_pred)
        f.write(str(seq_id.item())+","+str(ade)+","+str(fde)+","+str(maneuver_label)+","+str(maneuver_pred)+"\n")

        trajectories_plot(image, outputs_mean, keypoints, seq_id.item(), image_agent, centroid_current, ade, fde, history_traj, maneuver_label, maneuver_pred, outputs_list, outputs_var_mean, final_index)


cm = confusion_matrix(maneuver_gt, maneuver_outputs)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm.diagonal())

f.write("total_ADE =" + str(sum(ade_list)/len(ade_list)) +"\n")
f.write("total_FDE =" + str(sum(fde_list)/len(fde_list)) +"\n")

if config.MANEUVER_PRESENT:
    f.write("Straight_turn_accuracy =" + str(cm.diagonal()[0]) +"\n")
    f.write("Right_turn_accuracy =" + str(cm.diagonal()[1]) +"\n")
    f.write("Left_turn_accuracy =" + str(cm.diagonal()[2]) +"\n")
    f.write("Maneuver_classification =" + str(accuracy_score(maneuver_gt, maneuver_outputs)) +"\n")
f.close()

print("total_ADE", sum(ade_list)/len(ade_list))
print("total_FDE", sum(fde_list)/len(fde_list))

if config.MANEUVER_PRESENT:
    print("Straight_turn_accuracy", cm.diagonal()[0])
    print("Right_turn_accuracy", cm.diagonal()[1])
    print("Left_turn_accuracy", cm.diagonal()[2])
    print("Maneuver_classification", accuracy_score(maneuver_gt, maneuver_outputs))


