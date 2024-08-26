import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50
from tqdm import tqdm

import config

# Model definition
def build_model() -> torch.nn.Module:
    model = resnet50(pretrained=True)
    history_number_frames = 9
    num_in_channels = 3 + ((history_number_frames + 1) * 2)
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    num_targets = 2 * 50
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = build_model().to(device)
print(net)

# Transform the dataset
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
from data_load import FacialKeypointsDataset

'''
# create the transformed dataset
train_transformed_dataset = FacialKeypointsDataset(csv_file='/home/nesli/keypointdetection/input/lane_change_23_88456.csv',
                                        root_dir='/home/nesli/keypointdetection/input/training/')

print('Number of training images: ', len(train_transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = train_transformed_dataset[i]
    print(i, (sample['image']).size)
    #print(i, sample['image'], sample['keypoints'])

train_batch_size = 1
train_loader = DataLoader(train_transformed_dataset, 
                        batch_size=train_batch_size,
                        shuffle=True, 
                        num_workers=4)
'''

dataloader = FacialKeypointsDataset(csv_file='/home/nesli/keypointdetection/input/lane_change_23_88456.csv',
                                        root_dir='/home/nesli/keypointdetection/input/training/')
print('Number of training images: ', len(dataloader))

import torch.optim as optim
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters())

'''
# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='/home/nesli/keypointdetection/input/test_frames_keypoints.csv',
                                             root_dir='/home/nesli/keypointdetection/input/test/')
print('Number of test images: ', len(test_dataset))

for i in range(4):
    sample = test_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())

# load test data in batches
batch_size = 20
test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)
'''
'''
# training function
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        #image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        image, keypoints = data['image'], data['keypoints']
        # flatten the keypoints
        #keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image).reshape(keypoints.shape)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss
'''
# training function
def fit(model, dataloader):
    #dataloader.reset_batch_pointer(valid=False) #what is the purpose?
    train_running_loss = 0
    counter = 0
    for batch in range(dataloader.num_batches):
        counter +=1
        # Get batch data
        x, im = dataloader.next_batch()

        # For each sequence
        for sequence in range(dataloader.batch_size):
            # Get the data corresponding to the current sequence
            x_seq, im_seq = x[sequence], im[sequence]
            image = torch.tensor(im_seq, dtype=torch.float).to(config.DEVICE)
            keypoints = torch.tensor(x_seq, dtype=torch.float).to(config.DEVICE)
            #image, keypoints = data['image'].to(config.DEVICE), data['keypoints']
            image = image.permute(0, 3, 1, 2)
            # flatten the keypoints
            #keypoints = keypoints.view(keypoints.size(0), -1)
            optimizer.zero_grad()
            outputs = model(image) #.reshape(keypoints.shape)
            loss = criterion(outputs, keypoints)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss


'''
# validation function
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
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if (epoch+1) % 1 == 0 and i == 0:
                valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss
'''
train_loss = []
val_loss = []
n_epochs = 30
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1} of {n_epochs}")
    #train_epoch_loss = fit(net, train_loader, train_transformed_dataset)
    train_epoch_loss = fit(net, dataloader)
    #val_epoch_loss = validate(net, test_loader, test_dataset, epoch)
    train_loss.append(train_epoch_loss)
    #val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    #print(f'Val Loss: {val_epoch_loss:.4f}')


'''
from utilsslstm import DataLoader
image_data_dir= '/home/nesli/keypointdetection/input/training/'
batch_size = 8
seq_length = 1
datasets = [0]
train_loader = DataLoader(batch_size, seq_length, datasets, True, False)

import torch.optim as optim
#from utilsfld import valid_keypoints_plot

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters())
'''




'''
# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='/home/nesli/keypointdetection/input/test_frames_keypoints.csv',
                                             root_dir='/home/nesli/keypointdetection/input/test/')
print('Number of test images: ', len(test_dataset))

for i in range(4):
    sample = test_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())

# load test data in batches
batch_size = 20
test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)
'''
'''
def net_sample_output():
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']
        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)
        # forward pass to get net output
        output_pts = net(images)
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())

#Visualize test samples missing!
'''
'''


def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    """
    This function plots the regressed (predicted) keypoints and the actual 
    keypoints after each validation epoch for one image in the batch.
    """
    # detach the image, keypoints, and output tensors from GPU to CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()
    # just get a single datapoint from each batch
    img = image[0]
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]
    img = np.array(img, dtype='float32')
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
        plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'b.')
    plt.savefig(f"{config.OUTPUT_PATH}/val_epoch_{epoch}.png")
    plt.close()


# training function
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/counter
    return train_loss

# validation function
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
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if (epoch+1) % 1 == 0 and i == 0:
                valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

train_loss = []
val_loss = []
n_epochs = 30
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1} of {n_epochs}")
    train_epoch_loss = fit(net, train_loader, train_transformed_dataset)
    val_epoch_loss = validate(net, test_loader, test_dataset, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')



# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()
torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model.pth")
print('DONE TRAINING')

'''
'''
train_loss = []
val_loss = []
def train_net(n_epochs):

    # prepare the net for training
    net.train()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1} of {n_epochs}")
        running_loss = 0.0
        counter = 0
        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            counter +=  1
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']
            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            # forward pass to get outputs
            output_pts = net(images)
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()
            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
        train_loss = running_loss/counter
        #train_loss.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
    print('Finished Training')

train_net(n_epochs=30)
'''
