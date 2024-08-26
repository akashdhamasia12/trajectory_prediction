import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    #def __init__(self, csv_file, root_dir, transform=None):
    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.key_pts_frame = pd.read_csv(csv_file)
        self.data = np.genfromtxt(csv_file, delimiter=',')
        self.root_dir = root_dir
        self.resize = 224
        # Store the arguments
        self.batch_size = 1 #batch_size
        self.seq_length = 10 #seq_length
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row containing pedID, x, y
        self.all_frame_data = []
        # Each list would contain the frameIds of all the frames in the dataset
        self.frameList_data = []
        # Each list would contain the number of pedestrians in each frame in the dataset
        self.numPeds_data = []
        # Index of the current dataset
        dataset_index = 0
        # Frame IDs of the frames in the current dataset
        frameList = np.unique(self.data[0, :]).tolist()
        #numFrames = len(frameList)

        # Add the list of frameIDs to the frameList_data
        self.frameList_data.append(frameList)
        # Initialize the list of numPeds for the current dataset
        self.numPeds_data.append([])
        #self.numPeds_data = []
        # Initialize the list of numpy arrays for the current dataset
        self.all_frame_data.append([])
        #self.all_frame_data = []


        for ind, frame in enumerate(frameList):
            # Extract all pedestrians in current frame
            pedsInFrame = self.data[:, self.data[0, :] == frame]
            # Extract peds list
            pedsList = pedsInFrame[1, :].tolist()
            # Add number of peds in the current frame to the stored data
            self.numPeds_data[dataset_index].append(len(pedsList))
            # Initialize the row of the numpy array
            pedsWithPos = []
            # For each ped in the current frame
            for ped in pedsList:
                # Extract their x and y positions
                current_x = pedsInFrame[2, pedsInFrame[1, :] == ped][0] #originally 3
                current_y = pedsInFrame[3, pedsInFrame[1, :] == ped][0] #originally 2                
                # Add their pedID, x, y to the row of the numpy array
                pedsWithPos.append([ped, current_x, current_y])
            
            self.all_frame_data[dataset_index].append(np.array(pedsWithPos))
        dataset_index += 1

        print('Training data from dataset', ':', len(self.all_frame_data[0]))
        counter = int(len(self.all_frame_data[0]) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        print('Total number of training batches:', self.num_batches * 2) #Why do we multiply with 2?
        self.num_batches = self.num_batches * 2
        
        # Go to the first frame of the first dataset
        self.reset_batch_pointer()

    def __len__(self):

        # Frame IDs of the frames in the current dataset
        frameList = np.unique(self.data[0, :]).tolist()
        
        return len(frameList)
        #return len(self.key_pts_frame)

    #def __getitem__(self, idx):

    def reset_batch_pointer(self):
        '''
        Reset all pointers
        '''
        # Go to the first frame of the first dataset
        self.dataset_pointer = 0
        self.frame_pointer = 0

    def tick_batch_pointer(self):
        '''
        Advance the dataset pointer
        '''
        # Go to the next dataset
        self.dataset_pointer += 1
        # Set the frame pointer to zero for the current dataset
        self.frame_pointer = 0
        # If all datasets are done, then go to the first one again
        if self.dataset_pointer >= len(self.all_frame_data): #Bu kismi kontrol et!
            self.dataset_pointer = 0

    def next_batch(self, randomUpdate=True):

        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Image data
        image_batch = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            #frame_data = self.all_frame_data[self.dataset_pointer]
            #frameID = self.frameList_data[self.dataset_pointer]
            frame_data = self.all_frame_data[0]
            frameID = self.frameList_data[0]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]
                seq_source_frameID_data = frameID[idx:idx+self.seq_length] 

                st_image = np.zeros([224,224,3*len(seq_source_frameID_data)])
                for i in range(len(seq_source_frameID_data)):
                    image_name = str(int(seq_source_frameID_data[i])) + '_full.png'
                    image_name_full = os.path.join(self.root_dir,
                                            image_name)
                    
                    #image = mpimg.imread(image_name)
                    image = cv2.imread(image_name_full)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    orig_h, orig_w, channel = image.shape
                    image = cv2.resize(image, (224, 224))
                    image = image / 255.0

                    st_image[:,:, 3*(i):(3*(i+1))] = image

                print(st_image.shape)
                    
                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                image_batch.append(st_image)
                
                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length
                i += 1

            else:
                self.tick_batch_pointer()
        
        return x_batch, image_batch
        
        '''
        return {
            #'image': torch.tensor(image_batch, dtype=torch.float),
            #'keypoints': torch.tensor(x_batch, dtype=torch.float),
            'image': torch.tensor(image_batch, dtype=torch.float),
            'keypoints': x_batch,
        }
        '''


        #image_name = os.path.join(self.root_dir,
                                #self.key_pts_frame.iloc[idx, 0])
        
        #image = mpimg.imread(image_name)

        '''
        image_name_full = '/home/nesli/keypointdetection/lane_change_23_88456/' + '88388_full.png'
        image = cv2.imread(image_name_full)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, channel = image.shape
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        #plt.imshow(image)
        #plt.show()

        image_name_ego = '/home/nesli/keypointdetection/lane_change_23_88456/' + '88388_ego.png'
        image1 = cv2.imread(image_name_ego)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = cv2.resize(image1, (224, 224))
        image1 = image1 / 255.0
        image1 = np.expand_dims(image1, axis=2)
        orig_h1, orig_w1, channel1 = image1.shape

        image_name_agents = '/home/nesli/keypointdetection/lane_change_23_88456/' + '88388_agents.png'
        image2 = cv2.imread(image_name_agents)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image2 = cv2.resize(image2, (224, 224))
        image2 = image2 / 255.0
        image2 = np.expand_dims(image2, axis=2)
        orig_h2, orig_w2, channel2 = image2.shape

        #vol = np.stack([image, image1, image2])
        #vol.shape

        st_image = np.zeros([224,224,5])
        for i in range(3):
            if i < 3:
                st_image[:,:,i] = image[:,:,i] 
            elif i==3:
                st_image[:,:,i] = image1[:,:,1]
            elif i==4:
                st_image[:,:,i] = image2[:,:,1]
        '''

        '''
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        orig_h, orig_w, channel = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))

        #keypoints
        #key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = self.key_pts_frame.iloc[idx, 1:]
        key_pts = np.array(key_pts, dtype='float32')
        # reshape the keypoints
        key_pts = key_pts.reshape(-1, 2)
        # rescale keypoints according to image resize
        key_pts = key_pts * [self.resize / orig_w, self.resize / orig_h]

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(key_pts, dtype=torch.float),
        }
        '''