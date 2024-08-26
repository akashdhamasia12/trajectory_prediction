import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer
from ysdc_dataset_api.utils import transform_2d_points
from torch.utils.data import Dataset, DataLoader
import config
from tqdm import tqdm
import torch
import pickle

import matplotlib.image as Image

# Define a renderer config
renderer_config = {
    # parameters of feature maps to render
    'feature_map_params': {
        'rows': 224,
        'cols': 224,
        'resolution': 0.5,  # number of meters in one pixel
    },
    'renderers_groups': [
        # Having several feature map groups
        # allows to independently render feature maps with different history length.
        # This could be useful to render static features (road graph, etc.) once.
        {
            # start: int, first timestamp into the past to render, 0 – prediction time
            # stop: int, last timestamp to render inclusively, 24 – farthest known point into the past
            # step: int, grid step size,
            #            step=1 renders all points between start and stop,
            #            step=2 renders every second point, etc.
            'time_grid_params': {
                'start': 0,
                'stop': 10,
                'step': 1,
            },
            'renderers': [
                # each value is rendered at its own channel
                # occupancy -- 1 channel
                # velocity -- 2 channels (x, y)
                # acceleration -- 2 channels (x, y)
                # yaw -- 1 channel
                {'vehicles': ['occupancy']},
                # only occupancy and velocity are available for pedestrians
                {'pedestrians': ['occupancy']},
            ]
        },
        {
            'time_grid_params': {
                'start': 0,
                'stop': 0, #they are all same
                'step': 1,
            },
            'renderers': [
                {
                    'road_graph': [
                        'crosswalk_occupancy',
                        # 'crosswalk_availability',
                        # 'lane_availability',
                        # 'lane_direction',
                        'lane_occupancy',
                        # 'lane_priority',
                        # 'lane_speed_limit',
                        'road_polygons',
                    ]
                }
            ]
        }
    ]
}


# Create a renderer instance

renderer = FeatureRenderer(renderer_config)

# Load protobufs for training dataset.
dataset_path_train = "/home/akash/VehicleMotionPrediction/shifts/sdc/data/train_pb/"
dataset_path_val = "/home/akash/VehicleMotionPrediction/shifts/sdc/data/development_pb/"

# Path to file with training scene tags. Tags file is located in the raw data archive.
scene_tags_fpath_train = "/home/akash/VehicleMotionPrediction/shifts/sdc/data/train_tags.txt"
scene_tags_fpath_val = "/home/akash/VehicleMotionPrediction/shifts/sdc/data/development_tags.txt"


# # # To filter scenes by tags one should specify a filter function
# # # The scene tags dict has following structure:
# # # {
# # #     'day_time': one of {'kNight', 'kMorning', 'kAfternoon', 'kEvening'}
# # #     'season': one of {'kWinter', 'kSpring', 'kSummer', 'kAutumn'}
# # #     'track': one of {'Moscow' , 'Skolkovo', 'Innopolis', 'AnnArbor', 'Modiin', 'TelAviv'}
# # #     'sun_phase': one of {'kAstronomicalNight', 'kTwilight', 'kDaylight'}
# # #     'precipitation': one of {'kNoPrecipitation', 'kRain', 'kSleet', 'kSnow'}
# # # }
# # # Full description of protobuf message is available at tags.proto file in sources


def filter_scene(scene_tags_dict):
    if scene_tags_dict['track'] == 'Innopolis' and scene_tags_dict['precipitation'] == 'kRain':
        return True
    else:
        return False

def filter_moscow_no_precipitation_data(scene_tags_dict):
    if scene_tags_dict['track'] == 'Moscow':
        return True
    else:
        return False

def filter_ood_validation_data(scene_tags_dict):
    if scene_tags_dict['track'] in ['Skolkovo', 'Modiin', 'Innopolis', 'TelAviv']:
        return True
    else:
        return False
# Trajectory tags list can include any number of the following non-mutually exclusive tags.
# [
#     'kMoveLeft', 'kMoveRight', 'kMoveForward', 'kMoveBack',
#     'kAcceleration', 'kDeceleration', 'kUniform',
#     'kStopping', 'kStarting', 'kStationary'
# ]

def filter_trajectory(trajectory_tags_list):
    if 'kMoveRight' in trajectory_tags_list:
        return True
    else:
        return False

# Let's try to filter scenes.
# We need to use development data as long as train data contains no scenes with precipitation.

dataset_train_in = MotionPredictionDataset(
    dataset_path=dataset_path_train,
    scene_tags_fpath=scene_tags_fpath_train,
    feature_producer=renderer,
    # prerendered_dataset_path='/home/akash/VehicleMotionPrediction/shifts/sdc/data/train_rendered/',
    transform_ground_truth_to_agent_frame=True,
    scene_tags_filter=filter_moscow_no_precipitation_data,
    trajectory_tags_filter=filter_trajectory,
)

dataset_val_in = MotionPredictionDataset(
    dataset_path=dataset_path_val,
    scene_tags_fpath=scene_tags_fpath_val,
    feature_producer=renderer,
    # prerendered_dataset_path='/home/akash/VehicleMotionPrediction/shifts/sdc/data/development_rendered/',
    transform_ground_truth_to_agent_frame=True,
    scene_tags_filter=filter_moscow_no_precipitation_data,
    trajectory_tags_filter=filter_trajectory,
)

dataset_iter_train = iter(dataset_val_in)

agent_channels_ = 11
for i in range(1):
    data_item = next(dataset_iter_train)
    data_item = next(dataset_iter_train)
    data_item = next(dataset_iter_train)
    data_item = next(dataset_iter_train)

    print(f'{data_item.keys()}')
    print(data_item['feature_maps'].shape)

    im = data_item['feature_maps'].transpose(1, 2, 0)
    
    for i in range(agent_channels_):
        im_agent = im[:,:,i:i+1] #from current frames to past frames
        im_target = im[:,:,i+agent_channels_:i+agent_channels_+1] #from current frame to past frame

        name_agent = "shift_dataset/" + str(i) + "_agents.png"
        name_target = "shift_dataset/" + str(i) + "_targets.png"

        im_agent = im_agent[:,:,-1]
        im_target = im_target[:,:,-1]

        Image.imsave(name_agent, im_agent)
        Image.imsave(name_target, im_target)

    im_rgb = im[:,:,22:25]
    name_ = "shift_dataset/" + "map.png"
    Image.imsave(name_, im_rgb)

    im_rgb_ch1 = im[:,:,22:23]
    im_rgb_ch2 = im[:,:,23:24]
    im_rgb_ch3 = im[:,:,24:25]

    im_rgb_ch1 = im_rgb_ch1[:,:,-1]
    im_rgb_ch2 = im_rgb_ch2[:,:,-1]
    im_rgb_ch3 = im_rgb_ch3[:,:,-1]

    name_1 = "shift_dataset/map_ch1.png"
    Image.imsave(name_1, im_rgb_ch1,cmap='gray')
    name_2 = "shift_dataset/map_ch2.png"
    Image.imsave(name_2, im_rgb_ch2,cmap='gray')
    name_3 = "shift_dataset/map_ch3.png"
    Image.imsave(name_3, im_rgb_ch3,cmap='gray')

    #saving greyscale also
    R, G, B = im_rgb[:,:,0], im_rgb[:,:,1], im_rgb[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    name_ = "shift_dataset/map_grey.png"
    Image.imsave(name_, imgGray, cmap='gray')


#     # Plot vehicles occupancy, pedestrian occupancy, lane occupancy and road polygon
#     plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][0], origin='lower')
#     # plt.imshow(data_item['feature_maps'][1], origin='lower')
#     # plt.imshow(data_item['feature_maps'][2], origin='lower')
#     # plt.imshow(data_item['feature_maps'][3], origin='lower')
#     # plt.imshow(data_item['feature_maps'][4], origin='lower')
#     # plt.imshow(data_item['feature_maps'][5], origin='lower')
#     # plt.imshow(data_item['feature_maps'][6], origin='lower')
#     # plt.imshow(data_item['feature_maps'][7], origin='lower')
#     # plt.imshow(data_item['feature_maps'][8], origin='lower')
#     plt.imshow(data_item['feature_maps'][0], origin='lower', cmap='binary', alpha=0.7)
#     plt.imshow(data_item['feature_maps'][10], origin='lower', cmap='binary', alpha=0.5)

#     plt.figure(figsize=(10, 10))
#     plt.imshow(data_item['feature_maps'][24], origin='lower', cmap='binary', alpha=0.2)
#     plt.imshow(data_item['feature_maps'][27], origin='lower', cmap='binary', alpha=0.1)

#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][32], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][35], origin='lower', cmap='binary', alpha=0.2)

#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][40], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][43], origin='lower', cmap='binary', alpha=0.2)
 
#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][48], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][51], origin='lower', cmap='binary', alpha=0.2)
 
#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][56], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][59], origin='lower', cmap='binary', alpha=0.2)

#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][64], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][67], origin='lower', cmap='binary', alpha=0.2)

#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][72], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][75], origin='lower', cmap='binary', alpha=0.2)

#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][80], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][83], origin='lower', cmap='binary', alpha=0.2)

#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][88], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][91], origin='lower', cmap='binary', alpha=0.2)

#     # plt.figure(figsize=(10, 10))
#     # plt.imshow(data_item['feature_maps'][96], origin='lower', cmap='binary', alpha=0.2)
#     # plt.imshow(data_item['feature_maps'][99], origin='lower', cmap='binary', alpha=0.2)

#     plt.show()

# # # plt.figure(figsize=(10, 10))
# # # plt.imshow(data_item['feature_maps'][6], origin='lower', cmap='binary', alpha=0.2)

# # # plt.figure(figsize=(10, 10))
# # # plt.imshow(data_item['feature_maps'][0], origin='lower', cmap='binary', alpha=0.2)
