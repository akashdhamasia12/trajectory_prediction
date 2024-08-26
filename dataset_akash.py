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

# def train_test_split(csv_path, split):
#     df_data = pd.read_csv(csv_path)
#     len_data = len(df_data)
#     # calculate the validation data sample length
#     valid_split = int(len_data * split)
#     # calculate the training data samples length
#     train_split = int(len_data - valid_split)
#     training_samples = df_data.iloc[:train_split][:]
#     valid_samples = df_data.iloc[-valid_split:][:]
#     return training_samples, valid_samples

def train_test_split(sequences, split):

    len_data = len(sequences)
    # calculate the validation data sample length
    valid_split = int(len_data * split)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    training_samples = sequences[:train_split]
    valid_samples = sequences[-valid_split:]
    return training_samples, valid_samples

#we can add loops in case the frames are not enough
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

        image_path = f"{self.path}/training/"
        keypoints = []

        #stacking all images
        # for i in range(len(self.data[index][1])):
        for i in range(past_trajectory):

            full_image = image_path + str(int(self.data[index][1][i][0])) + "_full.png" #RGB image
            agents_image = image_path + str(int(self.data[index][1][i][0])) + "_agents.png" #agent channel
            ego_image = image_path + str(int(self.data[index][1][i][0])) + "_ego.png"

            full_i = cv2.imread(full_image)
            agent_i = cv2.imread(agents_image)
            ego_i = cv2.imread(ego_image)

            full_i = cv2.cvtColor(full_i, cv2.COLOR_BGR2RGB)
            agent_i = cv2.cvtColor(agent_i, cv2.COLOR_BGR2GRAY)
            ego_i = cv2.cvtColor(ego_i, cv2.COLOR_BGR2GRAY)

            full_i = cv2.resize(full_i, (self.resize, self.resize))
            agent_i = cv2.resize(agent_i, (self.resize, self.resize))
            agent_i = np.expand_dims(agent_i, axis=2)
            ego_i = cv2.resize(ego_i, (self.resize, self.resize))
            ego_i = np.expand_dims(ego_i, axis=2)

            result = np.concatenate((full_i, ego_i), axis=2)
            bev_frame = np.concatenate((result, agent_i), axis=2)


            # Normalising  (need to do in between -1 to 1)
            bev_frame = bev_frame / 255.0
            # bev_frame = np.expand_dims(bev_frame, axis=3)

            if i == 0:
                final_frame = np.copy(bev_frame)
            else:
                # which one should be appended first? final_frame or bev_frame?
                final_frame = np.concatenate((final_frame, bev_frame), axis=2)


        for j in range(past_trajectory, sequence_length):
            keypoints.append(self.data[index][1][j][1]) #x_cordinate
            keypoints.append(self.data[index][1][j][2]) #y_cordinate


        final_frame = np.transpose(final_frame) #image = np.transpose(image, (2, 0, 1))
        # print(final_frame.shape)
        keypoints = np.array(keypoints)
        # print(keypoints)

        return {
            'image': torch.tensor(final_frame, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }

        # image = cv2.imread(f"{self.path}/{self.data.iloc[index][0]}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # orig_h, orig_w, channel = image.shape
        # # resize the image into `resize` defined above
        # image = cv2.resize(image, (self.resize, self.resize))
        # # again reshape to add grayscale channel format
        # image = image / 255.0
        # # transpose for getting the channel size to index 0
        # image = np.transpose(image, (2, 0, 1))
        # # get the keypoints
        # keypoints = self.data.iloc[index][1:]
        # keypoints = np.array(keypoints, dtype='float32')
        # # reshape the keypoints
        # keypoints = keypoints.reshape(-1, 2)
        # # rescale keypoints according to image resize
        # keypoints = keypoints * [self.resize / orig_w, self.resize / orig_h]
        # return {
        #     'image': torch.tensor(image, dtype=torch.float),
        #     'keypoints': torch.tensor(keypoints, dtype=torch.float),
        # }
        

total_seq = create_sequences(f"{config.ROOT_PATH}/lane_change.csv")

# print(total_seq)
# get the training and validation data samples
# training_samples, valid_samples = train_test_split(f"{config.ROOT_PATH}/training_frames_keypoints.csv",
#                                                    config.TEST_SPLIT)

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

#Why did we add this part?
for i, data in tqdm(enumerate(valid_data)):
    print("akash")
    break
    # counter += 1
    # image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)

# # # whether to show dataset keypoint plots
# # if config.SHOW_DATASET_PLOT:
# #     utils.dataset_keypoints_plot(valid_data)
