import torch
from torch.autograd import Variable
import cv2
import pandas as pd
import numpy as np
import config
# import utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from os import path
import matplotlib.image as Image

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY

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
                if config.MANEUVER_PRESENT:
                    sequence.append([data[0][i+j], data[1][i+j], data[2][i+j], data[3][i+j], data[4][i+j], data[5][i+j]]) #frameid,x,y,avail,seqid,turn
                else:
                    sequence.append([data[0][i+j], data[1][i+j], data[2][i+j], data[3][i+j], data[4][i+j]]) #frameid,x,y,avail,seqid,turn
            all_seq.append([counter_i, sequence])
            counter_i=counter_i+1
        else:
            break
    print("number of total sequences", len(all_seq))

    shuffle_file = config.DATASET_PATH + "/shuffle_file.csv"

    if path.isfile(shuffle_file):
        random_agents = np.genfromtxt(shuffle_file, delimiter=',')
        selected_elements = [all_seq[int(index)] for index in random_agents]
    else:
        random_agents = np.random.choice(len(all_seq), size=len(all_seq), replace=False)
        selected_elements = [all_seq[index] for index in random_agents]
        np.savetxt(shuffle_file, random_agents, delimiter=",")

    return selected_elements


class TrajectoryDataset(Dataset):
    def __init__(self, samples, path, noise_):
        self.data = samples
        self.path = path
        self.resize = config.IMAGE_SIZE
        self.noise_ = noise_

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        image_path = f"{self.path}/train_images/"
        keypoints = []
        availability = []
        history_traj = []
        history_traj_availability = []

        # random_index = np.random.randint(0, past_trajectory, config.noise_freq)
        random_index = np.random.choice(range(past_trajectory), config.noise_freq, replace=False)

        if config.STACKING == "LYFT":
            #stacking all images - same as lyft (first all agents, then targets, then map) (from current frame to past frames)
            #Stack agent masks
            for i in range(0, past_trajectory):

                agents_image = image_path + str(int(self.data[index][1][i][4])) + "_" + str(int(self.data[index][1][i][0])) + "_agents.png"
                agent_i = cv2.imread(agents_image)
                agent_i = cv2.cvtColor(agent_i, cv2.COLOR_BGR2GRAY)
                agent_i = cv2.resize(agent_i, (self.resize, self.resize))

                if i == 0:
                    current_agent_i = np.copy(agent_i)
                    centroid_current = [self.data[index][1][0][1], self.data[index][1][0][2]]
                
                if self.noise_ == True and i in random_index:
                    (T, agent_i) = cv2.threshold(agent_i, 0, 255, cv2.THRESH_BINARY_INV)

                if config.NEIGHBOUR_RADIUS > 0:
                    mask = np.zeros(agent_i.shape[:2], dtype="uint8")
                    cv2.circle(mask, (int(self.data[index][1][i][1]/int(config.IMAGE_FACTOR)), int(self.data[index][1][i][2]/int(config.IMAGE_FACTOR))), int(config.NEIGHBOUR_RADIUS), 255, -1)
                    agent_i = cv2.bitwise_and(agent_i, agent_i, mask=mask)

                agent_i = np.expand_dims(agent_i, axis=2)

                if i == 0:
                    final_frame_agent = np.copy(agent_i)
                else:
                    final_frame_agent = np.concatenate((final_frame_agent, agent_i), axis=2)
    
            #Stack target masks
            for i in range(0, past_trajectory):

                ego_image = image_path + str(int(self.data[index][1][i][4])) + "_" + str(int(self.data[index][1][i][0])) + "_targets.png"
                ego_i = cv2.imread(ego_image)
                ego_i = cv2.cvtColor(ego_i, cv2.COLOR_BGR2GRAY)
                ego_i = cv2.resize(ego_i, (self.resize, self.resize))

                if self.noise_ == True and i in random_index:
                    (T, ego_i) = cv2.threshold(ego_i, 0, 255, cv2.THRESH_BINARY_INV)

                ego_i = np.expand_dims(ego_i, axis=2)

                if i == 0:
                    final_frame_ego = np.copy(ego_i)
                else:
                    final_frame_ego = np.concatenate((final_frame_ego, ego_i), axis=2)

            #Stack stacked-agents and stacked-ego frames
            final_frame_ae = np.concatenate((final_frame_agent, final_frame_ego), axis=2)

            #Stack map image with stacked frames
            full_image = image_path + str(int(self.data[index][1][past_trajectory-1][4])) + "_map.png" #RGB image
            full_i = cv2.imread(full_image)
            full_i = cv2.cvtColor(full_i, cv2.COLOR_BGR2RGB)
            full_i = cv2.resize(full_i, (self.resize, self.resize))
            final_frame = np.concatenate((final_frame_ae, full_i), axis=2)

            # Normalising image  (in between 0 to 1) (as in Resnet-50 we use Relu activation, normalization should be in between 0 to 1)
            final_frame = final_frame / 255.0

        else: 
            # stacking all images - different from lyft
            # for i in range(len(self.data[index][1])):
            for i in range(past_trajectory):

                agents_image = image_path + str(int(self.data[index][1][i][4])) + "_" + str(int(self.data[index][1][i][0])) + "_agents.png"
                ego_image = image_path + str(int(self.data[index][1][i][4])) + "_" + str(int(self.data[index][1][i][0])) + "_targets.png"

                agent_i = cv2.imread(agents_image)
                ego_i = cv2.imread(ego_image)

                agent_i = cv2.cvtColor(agent_i, cv2.COLOR_BGR2GRAY)
                ego_i = cv2.cvtColor(ego_i, cv2.COLOR_BGR2GRAY)
                #orig_h, orig_w = ego_i.shape

                agent_i = cv2.resize(agent_i, (self.resize, self.resize))

                if i == 0:
                    current_agent_i = np.copy(agent_i)
                    centroid_current = [self.data[index][1][0][1], self.data[index][1][0][2]]

                if config.NEIGHBOUR_RADIUS > 0:
                    mask = np.zeros(agent_i.shape[:2], dtype="uint8")
                    cv2.circle(mask, (int(self.data[index][1][i][1]/int(config.IMAGE_FACTOR)), int(self.data[index][1][i][2]/int(config.IMAGE_FACTOR))), int(config.NEIGHBOUR_RADIUS), 255, -1)
                    agent_i = cv2.bitwise_and(agent_i, agent_i, mask=mask)

                agent_i = np.expand_dims(agent_i, axis=2)
                ego_i = cv2.resize(ego_i, (self.resize, self.resize))
                ego_i = np.expand_dims(ego_i, axis=2)

                bev_frame = np.concatenate((agent_i, ego_i), axis=2) 
                
                if i == past_trajectory-1:
                    full_image = image_path + str(int(self.data[index][1][i][4])) + "_map.png" #RGB image
                    full_i = cv2.imread(full_image)
                    full_i = cv2.cvtColor(full_i, cv2.COLOR_BGR2RGB)
                    full_i = cv2.resize(full_i, (self.resize, self.resize))
                    bev_frame = np.concatenate((bev_frame, full_i), axis=2)

                # Normalising image  (in between 0 to 1) (as in Resnet-50 we use Relu activation, normalization should be in between 0 to 1)
                bev_frame = bev_frame / 255.0

                # Normalising image (in between -1 to 1)
                # min_range = -1
                # max_range = 1 
                # bev_frame_x = np.copy(bev_frame)
                # bev_std = (bev_frame_x - bev_frame_x.min()) / (bev_frame_x.max() - bev_frame_x.min())
                # bev_frame = bev_std * (max_range - min_range) + min_range

                # bev_frame = np.expand_dims(bev_frame, axis=3)

                if i == 0:
                    final_frame = np.copy(bev_frame)
                else:
                    final_frame = np.concatenate((final_frame, bev_frame), axis=2)


        for j in range(0, past_trajectory):
            history_traj.append(self.data[index][1][j][1]) #x_cordinate from current to past
            history_traj.append(self.data[index][1][j][2]) #y_cordinate
            history_traj_availability.append(self.data[index][1][j][3]) #availability for x           
            history_traj_availability.append(self.data[index][1][j][3]) #availability for y           


        #Labels generation
        for j in range(past_trajectory, sequence_length):

            keypoints.append(self.data[index][1][j][1]) #x_cordinate
            keypoints.append(self.data[index][1][j][2]) #y_cordinate
            availability.append(self.data[index][1][j][3]) #availability for x           
            availability.append(self.data[index][1][j][3]) #availability for y           

        if config.future_prediction > 0:
            keypoints = keypoints[0:2*config.future_prediction]
            availability = availability[0:2*config.future_prediction]

        final_frame = np.transpose(final_frame, (2, 0, 1)) #image = np.transpose(image, (2, 0, 1))
        #print(final_frame.shape)
        keypoints = np.array(keypoints) / int(config.IMAGE_FACTOR) #(keypoints containing x & y coordinates)
        history_traj = np.array(history_traj) / int(config.IMAGE_FACTOR) #(keypoints containing x & y coordinates)
        #print(keypoints.shape)
        availability = np.array(availability)
        #Normalise keypoints (in between -1 to 1)
        keypoints = ((keypoints/int(config.IMAGE_SIZE))*2) - 1
        if config.model_name == "lstm":
            history_traj = ((history_traj/int(config.IMAGE_SIZE))*2) - 1

        if config.MANEUVER_PRESENT:
            maneuver_label = int(self.data[index][1][i][5]) #frameid, x, y, avail, seq, turn
        else:
            maneuver_label = 0 #1:straight, 2: Right, 3:Left

        #generate weighing matrix
        if config.multimodal > 1:
            weights_mat1 = torch.empty(keypoints.shape[0]).fill_(0.5)
            weights_mat2 = torch.empty(keypoints.shape[0]).fill_(2)

            keypoints = np.tile(keypoints, int(config.multimodal))
            availability = np.tile(availability, int(config.multimodal))

            if maneuver_label == 1: #Straight driving
                weights_final = Variable(torch.cat((weights_mat2, weights_mat1, weights_mat1), 0))
            elif maneuver_label == 2: #Right driving
                weights_final = Variable(torch.cat((weights_mat1, weights_mat2, weights_mat1), 0))
            elif maneuver_label == 3: #Left driving
                weights_final = Variable(torch.cat((weights_mat1, weights_mat1, weights_mat2), 0))
            else:
                weights_final = torch.empty(keypoints.shape[0]).fill_(1)
        else:
            weights_final = torch.empty(keypoints.shape[0]).fill_(1)

        return {
            'image': torch.tensor(final_frame, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
            'availability': torch.tensor(availability, dtype=torch.float),
            'seq_id': torch.tensor(int(self.data[index][1][i][4]), dtype=torch.int),
            'current_agent_i': torch.tensor(current_agent_i, dtype=torch.float),
            'centroid_current': torch.tensor(centroid_current, dtype=torch.float),
            'history_traj': torch.tensor(history_traj, dtype=torch.float),
            'history_traj_availability': torch.tensor(history_traj_availability, dtype=torch.float),
            'maneuver_label': torch.tensor(maneuver_label, dtype=torch.int),
            'weights_final': weights_final
        }

total_seq = create_sequences(f"{config.DATASET_PATH}/train_csvs/train.csv")

num_of_sequences = int(len(total_seq) * config.num_sequences / 100)
print("num of sequences selected ", num_of_sequences)

training_samples, valid_samples = train_test_split(total_seq[:num_of_sequences], config.TEST_SPLIT)

#creating random training samples
size_training_random = len(training_samples)*1
random_agents = np.random.choice(len(training_samples), size=int(size_training_random), replace=False)
training_subset_random = [training_samples[index] for index in random_agents]

# initialize the dataset - `TrajectoryDataset()`
train_data = TrajectoryDataset(training_samples, 
                                 f"{config.DATASET_PATH}", False)

train_data_subset = TrajectoryDataset(training_subset_random, 
                                 f"{config.DATASET_PATH}", False)

valid_data = TrajectoryDataset(valid_samples, 
                                 f"{config.DATASET_PATH}", False)

valid_data_noise = TrajectoryDataset(valid_samples, 
                                 f"{config.DATASET_PATH}", True)

# test_data = TrajectoryDataset(total_seq, 
#                                  f"{config.DATASET_PATH}")

# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

valid_loader_noise = DataLoader(valid_data_noise, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

test_loader = DataLoader(valid_data, 
                          batch_size=1, 
                          shuffle=False)

test_loader_noise = DataLoader(valid_data_noise, 
                          batch_size=1, 
                          shuffle=False)

test_loader_training = DataLoader(train_data_subset, 
                          batch_size=1, 
                          shuffle=False)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")
print(f"Testing sample instances: {len(valid_data)}")
print(f"Testing sample training data instances: {len(train_data_subset)}")

# # # TO TEST
# for i, data in tqdm(enumerate(test_loader)):
#     image, keypoints, availability, seq_id, image_agent = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['seq_id'].to(config.DEVICE)), data['current_agent_i'].to(config.DEVICE)
#     # image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))


#     print(image.shape)
    # print(image_agent.shape)

    # print(keypoints.shape)

#     print(keypoints[0]," ", keypoints[1])
#     print(availability.shape)
#     print(seq_id.shape)
#     print(seq_id)
#     # keypoints = keypoints.view(keypoints.size(0), -1)
#     # print(keypoints.shape)
#     break

