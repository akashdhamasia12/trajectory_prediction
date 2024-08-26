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

total_sequences = 9000
val_sequences_n = int(total_sequences * config.TEST_SPLIT) 
train_sequences_n = total_sequences - val_sequences_n
print(train_sequences_n, " ", val_sequences_n )

class TrajectoryDataset(Dataset):
    def __init__(self, samples, path=""):
        self.data = samples
        self.path = path
        self.resize = config.IMAGE_SIZE

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        keypoints = []
        # availability = []
        # history_traj = []
        # history_traj_availability = []

        print(f'{self.data[index].keys()}')
        print(self.data[index]['feature_maps'].shape)


        return {
            'image': torch.tensor(keypoints, dtype=torch.float)
        }


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

dataset_val_out = MotionPredictionDataset(
    dataset_path=dataset_path_val,
    scene_tags_fpath=scene_tags_fpath_val,
    feature_producer=renderer,
    # prerendered_dataset_path='/home/akash/VehicleMotionPrediction/shifts/sdc/data/development_rendered/',
    transform_ground_truth_to_agent_frame=True,
    scene_tags_filter=filter_ood_validation_data,
    trajectory_tags_filter=filter_trajectory,
)

dataset_iter_train = iter(dataset_train_in)
dataset_iter_val_in = iter(dataset_val_in)
dataset_iter_val_out = iter(dataset_val_out)

train_sequences = []
val_in_sequences = []
val_out_sequences = []
count = 1

for i in range(train_sequences_n):
    train_sequences.append(next(dataset_iter_train))
    print(count)
    count = count + 1

for i in range(val_sequences_n):
    val_in_sequences.append(next(dataset_iter_val_in))
    print(count)
    count = count + 1

for i in range(val_sequences_n):
    val_out_sequences.append(next(dataset_iter_val_out))
    print(count)
    count = count + 1

with open('train_sequences.pkl', 'wb') as f:
    pickle.dump(train_sequences, f)

with open('val_in_sequences.pkl', 'wb') as f:
    pickle.dump(val_in_sequences, f)

with open('val_out_sequences.pkl', 'wb') as f:
    pickle.dump(val_out_sequences, f)


# with open('train_sequences.pkl', 'rb') as f:
#     train_sequences1 = pickle.load(f)

# with open('val_in_sequences.pkl', 'rb') as f:
#     val_in_sequences1 = pickle.load(f)

# with open('val_out_sequences.pkl', 'rb') as f:
#     val_out_sequences1 = pickle.load(f)


# # initialize the dataset - `TrajectoryDataset()`
# train_data = TrajectoryDataset(train_sequences1, 
#                                  f"{config.DATASET_PATH}")
# val_in_data = TrajectoryDataset(val_in_sequences1, 
#                                  f"{config.DATASET_PATH}")
# val_out_data = TrajectoryDataset(val_out_sequences1, 
#                                  f"{config.DATASET_PATH}")


# # prepare data loaders
# train_loader = DataLoader(train_data, 
#                           batch_size=config.BATCH_SIZE, 
#                           shuffle=True)

# valid_loader_in = DataLoader(val_in_data, 
#                           batch_size=config.BATCH_SIZE, 
#                           shuffle=False)

# valid_loader_out = DataLoader(val_out_data, 
#                           batch_size=config.BATCH_SIZE, 
#                           shuffle=False)

# test_loader = DataLoader(val_in_data, 
#                           batch_size=1, 
#                           shuffle=False)


# print(f"Training sample instances: {len(train_data)}")
# print(f"Validation sample instances: {len(val_in_data)}")
# print(f"Testing sample instances: {len(val_out_data)}")

# # TO TEST
# for i, data in tqdm(enumerate(test_loader)):
#     keypoints = data['image'].to(config.DEVICE)
#     # image, keypoints, availability, seq_id, image_agent = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['seq_id'].to(config.DEVICE)), data['current_agent_i'].to(config.DEVICE)
#     # # image, keypoints, availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE))


#     # print(image.shape)
#     # print(image_agent.shape)

#     # print(keypoints.shape)

#     # print(keypoints[0]," ", keypoints[1])
#     # print(availability.shape)
#     # print(seq_id.shape)
#     # print(seq_id)
#     # keypoints = keypoints.view(keypoints.size(0), -1)
#     # print(keypoints.shape)
#     break


# dataset_iter = iter(dataset)

# for i in range(10):
#     data_item = next(dataset_iter)

#     # print("ground_truth_trajectory", data_item['ground_truth_trajectory'])
#     transformed_gt = transform_2d_points(data_item['ground_truth_trajectory'], renderer.to_feature_map_tf)
#     # print("ground_truth_trajectory", transformed_gt)
#     transformed_gt = np.round(transformed_gt - 0.5).astype(np.int32)
#     # print("ground_truth_trajectory", transformed_gt)


#     plt.figure(figsize=(10, 10))
#     plt.imshow(data_item['feature_maps'][9], origin='lower', cmap='binary', alpha=0.7)
#     plt.imshow(data_item['feature_maps'][19], origin='lower', cmap='binary', alpha=0.5)
#     plt.imshow(data_item['feature_maps'][24], origin='lower', cmap='binary', alpha=0.2)
#     plt.imshow(data_item['feature_maps'][27], origin='lower', cmap='binary', alpha=0.1)
#     ax = plt.gca()
#     ax.add_collection(mc.LineCollection([transformed_gt], color='green'))

#     plt.show()



# # Number of scenes in dataset.
# # Actual number of objects in dataset is bigger,
# # since we consider multiple agents in a scene for prediction.
# print(dataset.num_scenes)

# # Create an iterator over the dataset
# dataset_iter = iter(dataset)

# for i in range(10):
#     data_item = next(dataset_iter)

#     print(f'{data_item.keys()}')

#     print(data_item['feature_maps'].shape)

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


# dataset = MotionPredictionDataset(
#     dataset_path=dataset_path,
#     scene_tags_fpath=scene_tags_fpath,
#     feature_producer=renderer,
#     transform_ground_truth_to_agent_frame=True,
# )

