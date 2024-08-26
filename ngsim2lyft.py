import os
import cv2
import matplotlib.image as Image
import numpy as np

# rootdir = '/dataset/NGSIM_CARLA_10Pred/results/'
# rootdir = '/dataset/NGSIM_CARLA/results/'
rootdir = 'dataset/NGSIM_CARLA/results/'

# total_seq_ = 100
# print("generating ", total_seq_, " sequences")

image_h_w = 300
print("NGSIM2LYFT")

history = 10 #(9 history + 1 current)
future_pred = 19 #8
# images_path_ = "/dataset/ngsim2lyft/train_images/"
# csvs_path_ = "/dataset/ngsim2lyft/train_csvs/"
# images_path_ = "/dataset/ngsim2lyft_10_19/train_images/"
# csvs_path_ = "/dataset/ngsim2lyft_10_19/train_csvs/"
images_path_ = "dataset/ngsim2lyft_subset/train_images/"
csvs_path_ = "dataset/ngsim2lyft_subset/train_csvs/"

train_csv = []
counter_seq = 0

for subdir, dirs, files in os.walk(rootdir):

    # if counter_seq == total_seq_:
    #     break

    if len(files) == 24:

        print("seq=", counter_seq)
        counter_frame = 0

        for i in range(0, history):
            agent_path = subdir + "/traffic_" + str(i) + ".png"
            target_path = subdir + "/ego_" + str(i) + ".png"
            im_agent = cv2.imread(agent_path)
            im_target = cv2.imread(target_path)

            if im_agent is not None and im_target is not None:
                name_agent = images_path_ + str(counter_seq) + "_" + str(counter_frame) + "_agents.png"
                name_target = images_path_ + str(counter_seq) + "_" + str(counter_frame) + "_targets.png"
                Image.imsave(name_agent, im_agent)
                Image.imsave(name_target, im_target)
                train_csv.append([counter_frame, 0, 0, 0, counter_seq]) #frameid, x, y, avail, seq
                counter_frame=counter_frame+1
            else:
                print("error")
                print(files)
                exit(0)

        map_path = subdir + "/map.png"
        im_map = cv2.imread(map_path)
        name_map = images_path_ + str(counter_seq) + "_map.png"
        Image.imsave(name_map, im_map)

        prediction_file = subdir + "/groundtruth.txt"
        data = np.genfromtxt(prediction_file, delimiter=',')
        # print(data.shape)

        prev_frame = 0

        for i in range(0, future_pred):

            if prev_frame == int(data[0][i+1]):
                availability = 0
            else:
                availability = 1

            y_frame =  image_h_w/2 - data[1][i+1] #x & y are reversed in NGSIM #i+1 because first frame is current frame
            x_frame =  image_h_w/2 - data[2][i+1]
            train_csv.append([counter_frame, x_frame, y_frame, availability, counter_seq]) #frameid, x, y, avail, seq
            counter_frame=counter_frame+1
            prev_frame = int(data[0][i+1])
        
        counter_seq=counter_seq+1

print("total_sequences", counter_seq)
csv_data_ = np.asarray(train_csv)
csv_train_path = csvs_path_ + "train.csv"
np.savetxt(csv_train_path, csv_data_.T, delimiter=",")

