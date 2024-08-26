import numpy as np
import config
from os import path
import pandas as pd

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


total_seq = create_sequences(f"{config.DATASET_PATH}/train_csvs/train.csv")

num_of_sequences = int(len(total_seq) * config.num_sequences / 100)
print("num of sequences selected ", num_of_sequences)

training_samples, valid_samples = train_test_split(total_seq[:num_of_sequences], config.TEST_SPLIT)

train_lstm_csv = []
test_lstm_csv = []
counter_frame = 0
counter_object = 0

for i in range(0, len(training_samples)):
    for j in range(0, len(training_samples[i][1])):

        y = ((training_samples[i][1][j][2]/int(config.IMAGE_SIZE))*2) - 1
        x = ((training_samples[i][1][j][1]/int(config.IMAGE_SIZE))*2) - 1
        # y = training_samples[i][1][j][2]
        # x = training_samples[i][1][j][1]
        train_lstm_csv.append([counter_frame, counter_object, y, x]) #LSTM: frameid, object_id, y, x
        counter_frame = counter_frame + 1
    counter_object = counter_object + 1

counter_frame = 0
counter_object = 0

for i in range(0, len(valid_samples)):
    for j in range(0, len(valid_samples[i][1])):
        y = ((valid_samples[i][1][j][2]/int(config.IMAGE_SIZE))*2) - 1
        x = ((valid_samples[i][1][j][1]/int(config.IMAGE_SIZE))*2) - 1
        # y = valid_samples[i][1][j][2]
        # x = valid_samples[i][1][j][1]
        test_lstm_csv.append([counter_frame, counter_object, y, x]) #LSTM: frameid, object_id, y, x
        counter_frame = counter_frame + 1
    counter_object = counter_object + 1

# csvs_path_train = "/home/adhamasia/Projects/social-lstm-quancore/data/train/lyft_9000/"
# csvs_path_test = "/home/adhamasia/Projects/social-lstm-quancore/data/test/lyft_9000/"

csvs_path_train = "/home/adhamasia/Projects/anirudh_org_org/data/lyft_9000/train/"
csvs_path_test = "/home/adhamasia/Projects/anirudh_org_org/data/lyft_9000/test/"

train_csv_data_ = np.asarray(train_lstm_csv)
csv_train_path = csvs_path_train + "pixel_pos_interpolate.csv" #"pixel_pos_interpolate.csv" #"train_lstm.txt"
# np.savetxt(csv_train_path, train_csv_data_.T, delimiter=",", fmt='%f')

# ['frame_num','ped_id','y','x']
test_csv_data_ = np.asarray(test_lstm_csv)
csv_test_path = csvs_path_test + "pixel_pos_interpolate.csv" #"pixel_pos_interpolate.csv" #"test_lstm.txt"


# df = pd.read_csv(directory, dtype={'frame_num':'int','ped_id':'int' }, delimiter = ' ',  header=None, names=column_names)

np.savetxt(csv_test_path, test_csv_data_.T, delimiter=",", fmt='%f')
np.savetxt(csv_train_path, train_csv_data_.T, delimiter=",", fmt='%f')

# column_names = ['frame_num','ped_id','y','x']
# df = pd.DataFrame(data=test_csv_data_)
# df1 = pd.DataFrame(data=train_csv_data_)

# df.to_csv(csv_test_path, header=None, index=None)
# df1.to_csv(csv_train_path, header=None, index=None)

# df.to_csv(csv_train_path)

# print(df)

# df = pd.read_csv(csv_test_path, dtype={'frame_num':'int','ped_id':'int' }, delimiter = ',',  header=None, names=column_names)
# print(df)