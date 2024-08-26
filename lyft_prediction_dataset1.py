import matplotlib.pyplot as plt
import matplotlib.image as Image
from matplotlib import collections  as mc

# import cv2

from torch.utils.data import DataLoader
import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os
import math

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

# set env variable for data
# os.environ["L5KIT_DATA_FOLDER"] = "/home/adhamasia/Datasets/lyft/lyft_prediction/"
# os.environ["L5KIT_DATA_FOLDER"] = "/mnt/raid0/trajectorypred/Datasets/Lyft/lyft_prediction/"
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/raid0/trajectorypred/Datasets/Lyft/lyft_prediction_new/"

# get config
cfg = load_config_data("./visualisation_config.yaml")
# cfg = load_config_data("./agent_motion_config.yaml")

print(cfg)

print(f'current raster_param:\n')
for k,v in cfg["raster_params"].items():
    print(f"{k}:{v}")

dm = LocalDataManager()
dataset_path = dm.require(cfg["train_data_loader"]["key"])
# dataset_path = dm.require(cfg["test_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

rasterizer = build_rasterizer(cfg, dm)
agent_train_dataset = AgentDataset(cfg, zarr_dataset, rasterizer)
# train_cfg = cfg["train_data_loader"]
# train_dataloader = DataLoader(agent_train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
#                              num_workers=train_cfg["num_workers"])

print("total_sequences", len(agent_train_dataset))

# '''
# tr_it = iter(train_dataloader)
# data = next(tr_it)
# inputs = data["image"]
# numpy_image = np.array(inputs)
# print(numpy_image.shape)

total_seq_ = 30000 #9000 #30000 #10000 (#10000 for Right, 10000 for Left, 10000 for straight)
print("generating ", total_seq_, " sequences")

num_history_channels_ = (cfg["model_params"]["history_num_frames"] + 1) * 2
agent_channels_ = int(num_history_channels_ / 2) #past
total_channels = num_history_channels_ + 3
future_pred = cfg["model_params"]["future_num_frames"]
# images_path_ = "/mnt/raid0/trajectorypred/train_images/"
# csvs_path_ = "/mnt/raid0/trajectorypred/train_csvs/"
images_path_ = "/mnt/raid0/trajectorypred/lyft_10_30_30000_balanced/train_images/"
csvs_path_ = "/mnt/raid0/trajectorypred/lyft_10_30_30000_balanced/train_csvs/"
# images_path_ = "/mnt/raid0/trajectorypred/test_images/"
# csvs_path_ = "/mnt/raid0/trajectorypred/test_csvs/"

train_csv = []
counter_seq = 0

# random_agents = np.random.randint(low=0, high=len(agent_train_dataset)-1, size=total_seq_) #it does repetition
random_agents = np.random.choice(len(agent_train_dataset), size=len(agent_train_dataset), replace=False) #similar to np.arange and shuffle (no repeat)

f=open(csvs_path_ + "agent_ids.txt","w")
# f_d=open("displacements.txt","w")

counter_straight = 0
counter_left = 0
counter_right = 0

for agent_id in random_agents:
    # agent_id = 9957612 
    print("seq=", counter_seq, ",", agent_id)

    if counter_seq == total_seq_:
        break

    counter_frame = 0
    data = agent_train_dataset[agent_id]
    im = data["image"].transpose(1, 2, 0)
    # targets = data["target_positions"]
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"]) #(30x2)
    targets_avail = data["target_availabilities"]

    history_positions_pixels = transform_points(data["history_positions"] + data["centroid"][:2], data["world_to_image"]) #(11x2)
    history_avail = data["history_availabilities"]

    displacement = np.linalg.norm(history_positions_pixels[0] - target_positions_pixels[future_pred-1]) 

    line_history = [history_positions_pixels[agent_channels_-1], history_positions_pixels[0]]
    line_future = [target_positions_pixels[0], target_positions_pixels[future_pred-1]]

    angle_btw_lines, angle_btw_lines_tan = ang(line_history, line_future)

    # print("displacement= ", displacement)
    # print("angle_btw_lines ", angle_btw_lines)
    # print("angle_btw_lines_tan ", angle_btw_lines_tan)
    # f.write(str(displacement)+"\n")

    #for Plotting
    # lines = [line_history, line_future]
    # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1)])
    # lc = mc.LineCollection(lines, colors=c, linewidths=2)
    # fig, ax = plt.subplots()
    # ax.add_collection(lc)
    # ax.set(xlim=(0, 224), ylim=(224, 0))
    # plt.show()

    if displacement >= 5 and angle_btw_lines < 5 and counter_straight < int(total_seq_/3): #almost straight
        counter_straight = counter_straight + 1
        turn = 1 #straight
    elif displacement >= 5 and angle_btw_lines_tan > 0 and angle_btw_lines > 5 and counter_right < int(total_seq_/3): #Right Turn
        counter_right = counter_right + 1
        turn = 2 #right
    elif displacement >= 5 and angle_btw_lines_tan < 0 and angle_btw_lines > 5 and counter_left < int(total_seq_/3): #Left Turn
        counter_left = counter_left + 1
        turn = 3 #left
    else:
        turn = None #None

    print(counter_straight, counter_right, counter_left)
#     # numpy_centroid = np.array(data["centroid"])
#     # centroid = np.reshape(numpy_centroid, (1,2))
#     # # centroid = transform_points(data["centroid"], data["world_to_image"])
#     # centroid_pixels = transform_points(centroid, data["world_to_image"])
#     # print("centroid", centroid_pixels)
 

    if turn:

        f.write(str(agent_id)+"\n")

        for i in range(agent_channels_):
            im_agent = im[:,:,i:i+1] #from current frames to past frames
            im_target = im[:,:,i+agent_channels_:i+agent_channels_+1] #from current frame to past frame

            name_agent = images_path_ + str(agent_id) + "_" + str(counter_frame) + "_agents.png"
            name_target = images_path_ + str(agent_id) + "_" + str(counter_frame) + "_targets.png"

            im_agent = im_agent[:,:,-1]
            im_target = im_target[:,:,-1]

            Image.imsave(name_agent, im_agent)
            Image.imsave(name_target, im_target)

            # plt.imshow(im_agent)
            # plt.savefig(name_agent)
            # plt.show()
    
            train_csv.append([counter_frame, history_positions_pixels[i][0], history_positions_pixels[i][1], history_avail[i], agent_id, turn]) #frameid, x, y, avail, seq
            counter_frame=counter_frame+1

        im_rgb = im[:,:,num_history_channels_:total_channels]
        name_ = images_path_ + str(agent_id) + "_map.png"
        Image.imsave(name_, im_rgb)

        for future in range(future_pred):
            train_csv.append([counter_frame, target_positions_pixels[future][0], target_positions_pixels[future][1], targets_avail[future], agent_id, turn]) #frameid, x, y, avail, seq
            counter_frame=counter_frame+1

        counter_seq=counter_seq+1

# f_d.close()

f.close()
csv_data_ = np.asarray(train_csv)
# csv_train_path = csvs_path_ + "train.csv"
csv_train_path = csvs_path_ + "train.csv"
np.savetxt(csv_train_path, csv_data_.T, delimiter=",")


# inputs = data["image"]
# numpy_image = np.array(inputs)
# print(numpy_image.shape)

# im = data["image"].transpose(1, 2, 0)
# numpy_image = np.array(im)
# print(numpy_image.shape)
# plt.title("sample")

# history_frames_ = 10
# total_channels = (history_frames_ + 1)*2 + 3
# agent_channels = history_frames_ + 1
# target_channels = history_frames_ + 1

# for i in range(agent_channels):
#     im_ = im[:,:,i:i+1]
#     name_ = "/home/adhamasia/agentss_" + str(i) + ".png"
#     plt.imshow(im_)
#     plt.savefig(name_)
#     # plt.imshow(im_)
#     # plt.show()

# for i in range(agent_channels,target_channels):
#     im_ = im[:,:,i:i+1]
#     name_ = "/home/adhamasia/targetss_" + str(i) + ".png"
#     plt.imshow(im_)
#     plt.savefig(name_)

# im_rgb = im[:,:,total_channels-3:total_channels]
# plt.imshow(im_rgb)
# plt.show()

# im1 = im[:,:,0:1] #agent_masks
# plt.imshow(im1)
# plt.show()

# im2 = im[:,:,1:2] #agent_masks
# plt.imshow(im2)
# plt.show()

# im3 = im[:,:,2:3] #target_masks
# plt.imshow(im3)
# plt.show()

# im4 = im[:,:,3:4] #target_masks
# plt.imshow(im4)
# plt.show()

# im5 = im[:,:,4:7] #rgb_masks
# plt.imshow(im5)
# plt.show()


# data = agent_train_dataset[90]
# im = data["image"].transpose(1, 2, 0)
# # plt.title("sample")
# # im = im[:,:,1:2]
# # im = im[:,:,-1]

# im = im[:,:,num_history_channels_:total_channels]


# numpy_image = np.array(im)
# # numpy_image = np.reshape
# # numpy_image = numpy_image[:,:,1:2]
# print(numpy_image.shape)

# # plt.savefig("test.png")
# # matplotlib.image.imsave('name.png', im, cmap="gray")
# matplotlib.image.imsave('name.png', im)
# matplotlib.image.imsave('name.png', numpy_image, cmap="gray")



# numpy_image = np.array(im)
# numpy_image = numpy_image[:,:,1:2]
# print(numpy_image.shape)

# print(numpy_image[:,:,:3].shape)
# im = agent_train_dataset.rasterizer.to_rgb(im)
# im = im[:,:,1:2]
# matplotlib.image.imsave('name.png', im)
# cv2.imwrite("filename.png", im)
# im1 = Image.fromarray(numpy_image)
# im1.save('test.png')

# print(len(im))
# plt.title("sample")
# plt.imshow(im)
# plt.show()
