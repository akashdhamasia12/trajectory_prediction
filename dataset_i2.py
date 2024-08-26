import torch
import cv2
import pandas as pd
import numpy as np
import config
# import utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


sequence_length = 50
past_trajectory = 10


def train_test_split(sequences, split):

    len_data = len(sequences)
    # calculate the validation data sample length
    valid_split = int(len_data * split)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    training_samples = sequences[:train_split]
    valid_samples = sequences[-valid_split:]
    return training_samples, valid_samples

def create_sequences(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    all_seq = []

    for i in range(data.shape[1]):
        sequence = []
        if (i + sequence_length) < data.shape[1]:
            for j in range(sequence_length):
                    sequence.append([data[0][i+j], data[1][i+j], data[2][i+j]])
            all_seq.append([i, sequence])
        else:
            break

    print("number of sequences", len(all_seq))
    return all_seq


class FaceKeypointDataset(Dataset):
    def __init__(self, samples, path):
        self.data = samples
        self.path = path
        self.resize = 224
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        image_path = f"{self.path}/input/training/"
        keypoints = []

        #stacking all images
        # for i in range(len(self.data[index][1])):
        for i in range(past_trajectory):

            map_image = image_path + str(int(self.data[index][1][i][0])) + "_full.png" #RGB image
            agents_image = image_path + str(int(self.data[index][1][i][0])) + "_agents.png" #agent channel
            ego_image = image_path + str(int(self.data[index][1][i][0])) + "_ego.png"

            map_i = cv2.imread(map_image)
            agent_i = cv2.imread(agents_image)
            ego_i = cv2.imread(ego_image)

            map_i = cv2.cvtColor(map_i, cv2.COLOR_BGR2RGB)
            agent_i = cv2.cvtColor(agent_i, cv2.COLOR_BGR2GRAY)
            ego_i = cv2.cvtColor(ego_i, cv2.COLOR_BGR2GRAY)

            map_i = cv2.resize(map_i, (self.resize, self.resize))
            agent_i = cv2.resize(agent_i, (self.resize, self.resize))
            agent_i = np.expand_dims(agent_i, axis=2)
            ego_i = cv2.resize(ego_i, (self.resize, self.resize))
            ego_i = np.expand_dims(ego_i, axis=2)
            
            result = np.concatenate((map_i, ego_i), axis=2)
            bev_frame = np.concatenate((result, agent_i), axis=2)

            print(bev_frame.shape)

            # Normalising  (need to do in between -1 to 1)
            bev_frame = bev_frame / 255.0

            if i == 0:
                prev_bev = bev_frame
            else:
                final_frame = np.concatenate((bev_frame, prev_bev), axis=2)
                prev_bev = bev_frame

        for j in range(past_trajectory, sequence_length):
            keypoints.append(self.data[index][1][j][1]) #x_cordinate
            keypoints.append(self.data[index][1][j][2]) #y_cordinate

        keypoints = np.array(keypoints)

        return {
            'image': torch.tensor(final_frame, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }


total_seq = create_sequences(f"{config.ROOT_PATH}/lane_change.csv")


training_samples, valid_samples = train_test_split(total_seq, config.TEST_SPLIT)


# initialize the dataset - `FaceKeypointDataset()`
train_data = FaceKeypointDataset(training_samples, 
                                 f"{config.ROOT_PATH}")
valid_data = FaceKeypointDataset(valid_samples, 
                                 f"{config.ROOT_PATH}")


# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")


