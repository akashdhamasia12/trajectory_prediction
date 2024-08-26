from typing_extensions import final
import config
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision.models.resnet import resnet50
from resnet_tnse import resnet50
import torch.nn as nn
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dataset_shift import test_loader_out, test_loader_in
from dataset import test_loader, test_loader_noise

import torchvision.models as models

import matplotlib.pyplot as plt
import matplotlib.image as Image
# from PIL import Image
import cv2
import math

from sklearn.manifold import TSNE

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
history_frames = past_trajectory*2 + 3
total_maneuvers = ["none", "straight", "right", "left"]

#set random seeds
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


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

counter_1 = 0


features = None
model.eval()

with torch.no_grad():

    for i, data in enumerate(test_loader):

        if config.DATASET == "SHIFT":
            image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))
        else:
            image, keypoints, availability, seq_id, image_agent, centroid_current, history_traj, history_traj_availability, maneuver_label = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['seq_id'].to(config.DEVICE)), torch.squeeze(data['current_agent_i'].to(config.DEVICE)), torch.squeeze(data['centroid_current'].to(config.DEVICE)), torch.squeeze(data['history_traj'].to(config.DEVICE)), torch.squeeze(data['history_traj_availability'].to(config.DEVICE)), torch.squeeze(data['maneuver_label'].to(config.DEVICE))

        # model = set_training_mode_for_dropout(model, True)
        # outputs, _ = model.forward(image)
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = torch.squeeze(model(image), 0) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = outputs.view(keypoints.shape[0], -1)

        current_features = outputs.detach().cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

        # counter_1 += 1
        # if counter_1 == 1000:
        #     break

counter_1 = 0

with torch.no_grad():

    for i, data in enumerate(test_loader_noise):

        image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))

        # model = set_training_mode_for_dropout(model, True)
        # outputs, _ = model.forward(image)
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = torch.squeeze(model(image), 0) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = outputs.view(keypoints.shape[0], -1)

        current_features = outputs.detach().cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

        # counter_1 += 1
        # if counter_1 == 1000:
        #     break

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# tsne = TSNE(n_components=2).fit_transform(features)

# # extract x and y coordinates representing the positions of the images on T-SNE plot
# tx = tsne[:, 0]
# ty = tsne[:, 1]

# # scale and move the coordinates so they fit [0; 1] range
# tx = scale_to_01_range(tx)
# ty = scale_to_01_range(ty)

# initialize matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# ax.scatter(tx, ty)

# # build a legend using the labels we set previously
# ax.legend(loc='best')

# # finally, show the plot
# plt.show()


####################################2nd model#########################################################################

model = build_model().to(config.DEVICE)
# print(model)

if config.MULTIPLE_GPU:
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=[config.cuda_device]).cuda()

model.to(config.DEVICE)

# # load the model checkpoint
print('Loading checkpoint')
checkpoint = torch.load("/home/akash/datasets/shift_10_25_9000/outputs_cnn_25_42/best_epoch_MSE_.pth")
# checkpoint = torch.load(config.model_path)
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs_uncertainty_all/best_epoch_MSE_.pth")
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs/best_epoch_MSE_.pth")

model_epoch = checkpoint['epoch']
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])#, strict=False) # Error(s) in loading state_dict for ResNet: -> added strict-False
print('Loaded checkpoint at epoch', model_epoch)

print('run')

counter_1 = 0
# features = None
model.eval()

with torch.no_grad():

    for i, data in enumerate(test_loader_in):

        image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))

        # model = set_training_mode_for_dropout(model, True)
        # outputs, _ = model.forward(image)
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = torch.squeeze(model(image), 0) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = outputs.view(keypoints.shape[0], -1)

        current_features = outputs.detach().cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

        # counter_1 += 1
        # if counter_1 == 1000:
        #     break

# features_out = None
counter_1 = 0

with torch.no_grad():

    for i, data in enumerate(test_loader_out):

        image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))

        # model = set_training_mode_for_dropout(model, True)
        # outputs, _ = model.forward(image)
        outputs = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = torch.squeeze(model(image), 0) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
        # outputs = outputs.view(keypoints.shape[0], -1)

        current_features = outputs.detach().cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

        # counter_1 += 1
        # if counter_1 == 1000:
        #     break



tsne = TSNE(n_components=2).fit_transform(features)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

# scale and move the coordinates so they fit [0; 1] range
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)


current_tx = tx[0:1800]
current_ty = ty[0:1800]
ax.scatter(current_tx, current_ty, c='b')

current_tx = tx[1800:3600]
current_ty = ty[1800:3600]
ax.scatter(current_tx, current_ty, c='g')

current_tx = tx[3600:5400]
current_ty = ty[3600:5400]
ax.scatter(current_tx, current_ty, c='r')

current_tx = tx[5400:]
current_ty = ty[5400:]
ax.scatter(current_tx, current_ty, c='y')

plt.show()
fig.savefig("tnse_visualization.png")


