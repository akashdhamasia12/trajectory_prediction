import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config
from tqdm import tqdm
import torch
import pickle

from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer
from ysdc_dataset_api.utils import transform_2d_points

# Define a renderer config
renderer_config = {
    'feature_map_params': {
        'rows': 224,
        'cols': 224,
        'resolution': 0.5,  # number of meters in one pixel
    },
    'renderers_groups': [
        {
            'time_grid_params': {
                'start': 0,
                'stop': 9,
                'step': 1,
            },
            'renderers': [
                {'vehicles': ['occupancy']},
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
                        'crosswalk_availability',
                        'lane_availability',
                        'lane_direction',
                        'lane_occupancy',
                        'lane_priority',
                        'lane_speed_limit',
                        'road_polygons',
                    ]
                }
            ]
        }
    ]
}

# Create a renderer instance
renderer = FeatureRenderer(renderer_config)

class TrajectoryDataset(Dataset):
    def __init__(self, samples, path=""):
        self.data = samples
        self.path = path
        self.resize = config.IMAGE_SIZE

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):


        final_frame = self.data[index]['feature_maps']
        # Normalising image  (in between 0 to 1) (as in Resnet-50 we use Relu activation, normalization should be in between 0 to 1)
        final_frame = final_frame / 255.0

        # print("ground_truth_trajectory", self.data[index]['ground_truth_trajectory'])
        transformed_gt = transform_2d_points(self.data[index]['ground_truth_trajectory'], renderer.to_feature_map_tf)
        transformed_gt = np.round(transformed_gt - 0.5).astype(np.int32)

        keypoints = transformed_gt.flatten()
        availability = np.ones(keypoints.shape[0])

        if config.future_prediction > 0:
            keypoints = keypoints[0:2*config.future_prediction]
            availability = availability[0:2*config.future_prediction]

        #Normalise keypoints (in between -1 to 1)
        keypoints = ((keypoints/int(config.IMAGE_SIZE))*2) - 1

        return {
            'image': torch.tensor(final_frame, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
            'availability': torch.tensor(availability, dtype=torch.float)
        }


with open('train_sequences.pkl', 'rb') as f:
    train_sequences1 = pickle.load(f)

with open('val_in_sequences.pkl', 'rb') as f:
    val_in_sequences1 = pickle.load(f)

with open('val_out_sequences.pkl', 'rb') as f:
    val_out_sequences1 = pickle.load(f)


# initialize the dataset - `TrajectoryDataset()`
train_data = TrajectoryDataset(train_sequences1, 
                                 f"{config.DATASET_PATH}")
valid_data = TrajectoryDataset(val_in_sequences1, 
                                 f"{config.DATASET_PATH}")
valid_data_out = TrajectoryDataset(val_out_sequences1, 
                                 f"{config.DATASET_PATH}")


# prepare data loaders
train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

valid_loader_out = DataLoader(valid_data_out, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

test_loader_in = DataLoader(valid_data, 
                          batch_size=1, 
                          shuffle=False)

test_loader_out = DataLoader(valid_data_out, 
                          batch_size=1, 
                          shuffle=False)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")
print(f"Testing sample instances: {len(valid_data_out)}")

# TO TEST
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
