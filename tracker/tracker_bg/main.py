import cv2
from tracker import *
import os
import numpy as np

################
import torch
import random
import os
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from torchvision.models.resnet import resnet50
from resnet50_dropout import resnet50
import torchvision.models as models

import config

# if config.DATASET == "SHIFT":
#     from dataset_shift import train_data, train_loader, valid_data, valid_loader, valid_loader_out
# else:
#     from dataset import train_data, train_loader, valid_data, valid_loader, valid_loader_noise

# from tqdm import tqdm
# import utils
# import robust_loss_pytorch

import copy
##############################

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
history_frames = past_trajectory*2 + 3
total_maneuvers = ["none", "straight", "right", "left"]

# Model Creation
def build_model() -> torch.nn.Module:
    # load pre-trained Conv2D model
    # '''
    model = resnet50(pretrained=True, p=config.dropout_prob)

    # change input channels number to match the rasterizer's output
    if config.DATASET == "SHIFT":
        num_in_channels = 25
    else:
        num_in_channels = 3 + (2*past_trajectory)

    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states

    if config.future_prediction > 0:
        num_targets = 2 * config.future_prediction
    else:
        num_targets = 2 * (sequence_length - past_trajectory)

    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

model = build_model().to(config.DEVICE)
# print(model)

if config.GPU:
    if config.MULTIPLE_GPU:
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=[config.cuda_device]).cuda()

model.to(config.DEVICE)


# # load the model checkpoint
print('Loading checkpoint')
checkpoint = torch.load(f"{config.OUTPUT_PATH}/{config.BEST_EPOCH}")
# checkpoint = torch.load(config.model_path)
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs_uncertainty_all/best_epoch_MSE_.pth")
# checkpoint = torch.load("/home/neslihan/akash/datasets/lyft_10_30_9000_balanced/outputs/best_epoch_MSE_.pth")

model_epoch = checkpoint['epoch']
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])#, strict=False) # Error(s) in loading state_dict for ResNet: -> added strict-False
print('Loaded checkpoint at epoch', model_epoch)

print('run')
model.eval()


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def main():
    # clear tracked objects from previous run
    if(os.path.exists("objects.csv")):
        os.remove("objects.csv")


    # Create tracker object
    tracker = EuclideanDistTracker()
    # https://cwwp2.dot.ca.gov/vm/streamlist.htm
    # cap = cv2.VideoCapture("rtsp://192.168.1.213:554/11")
    cap = cv2.VideoCapture("videoplayback.mp4")
    complete_map = cv2.imread("complete_map_.png")
    dim = (224*3, 224*3)
    complete_map = cv2.resize(complete_map, dim, interpolation =cv2.INTER_AREA)
    print(complete_map.shape)


    # Object detection from Stable camera
    object_detector = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=800, detectShadows=False)
    frame_track = []
    current_frame_id = 0
    # video=cv2.VideoWriter('video.mp4',-1,1,(224*2,224*2))

    counter = 0
    skip = 0
    prev_track_id = -1

    while True:
        ret, frame = cap.read()
        counter = counter + 1

        # if counter % skip != 0:
        #     continue

        for k in range(0, skip):
            ret, frame = cap.read()


        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        dim = (224, 224)
        frame = cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
        # frame = rescale_frame(frame, percent=25) #45
        height, width, _ = frame.shape
        # print(height, width)
        # cv2.imwrite("saved_image.png", frame)

        # cv2.imshow("Frame", frame)
        # cv2.imshow("map", complete_map)
        # key = cv2.waitKey(0)
        # exit(0)

        # Extract Region of interest
        roi = frame.copy()#[0: 224,0: 224]
    

        # 1. Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 400:
                # Draw contours if needed
                cv2.drawContours(roi, [cnt], -1, (0, 0, 255), )
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        boxes_ids = tracker.update(detections)

        # 2. Object Tracking
        if len(boxes_ids) > 0:

            boxes_ids_ = np.array(boxes_ids)
            frame_track.append(boxes_ids_)
            frame_track_ = np.array(frame_track)

            if prev_track_id in boxes_ids_[:,4]:
                prev_track_id = prev_track_id
            else:
                prev_track_id = -1

            if prev_track_id == -1:
                track_object_id = np.random.choice(boxes_ids_[:,4])
            else:
                track_object_id = prev_track_id

            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.putText(roi, "obj: " + str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # cv2.circle(roi, (x + int(w/2), y + int(h/2)), 2, (0, 0, 255), 2)
                if id == track_object_id:
                    cv2.circle(roi, (x + int(w/2), y + int(h/2)), 2, (0, 255, 0), 2)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    target_x_orig = x + int(w/2)            
                    target_y_orig = y + int(h/2)            
                else:
                    cv2.circle(roi, (x + int(w/2), y + int(h/2)), 2, (255, 0, 0), 2)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 3)


            target_masks = []
            agent_masks = []
            map_flag = False
            disp_target_x = 0 
            disp_target_y = 0 

            prev_track_id = track_object_id

            #generate map and mask image
            for history in range(0, 11):

                indexing_frame = current_frame_id - history

                if indexing_frame >= 0:
                    boxes_current_frame = frame_track_[indexing_frame]

                    # get_map = np.zeros((224,224,3), np.uint8)
                    get_index = np.where(boxes_current_frame[:,4] == track_object_id)
                    if np.size(get_index[0]):
                        x, y, w, h, id = boxes_current_frame[get_index[0][0]] 

                        mask_image_target = np.zeros((224,224,3), np.uint8)
                        mask_image_agent = np.zeros((224,224,3), np.uint8)

                        #generate map
                        if not map_flag:
                            center_cord_obj = (x + int(w/2), y + int(h/2))
                            map_center_cord = (224 + center_cord_obj[0], 224 + center_cord_obj[1]) #object in the complete map
                            get_map = np.copy(complete_map[map_center_cord[1]-112:map_center_cord[1]+112, map_center_cord[0]-112:map_center_cord[0]+112]) #get the patch of map from that object
                            get_map1 = np.copy(get_map)
                            past_x = x    
                            past_y = y    
                            map_flag = True


                        disp_target_x = past_x - x     
                        disp_target_y = past_y - y     

                        #generate mask_target
                        target_cord = (112-disp_target_x, 112-disp_target_y)
                        width_vehicle = (10, 5)

                        target_cord = np.clip(target_cord, 5, 215)                    
                        cv2.rectangle(mask_image_target, (target_cord[0]-int(width_vehicle[0]/2), target_cord[1]-int(width_vehicle[1]/2)), (target_cord[0] + int(width_vehicle[0]/2), target_cord[1] + int(width_vehicle[1]/2)), (0, 255, 0), -1)
                        cv2.circle(get_map1, (int(target_cord[0]), int(target_cord[1])), 1, (0, 255, 0), 1)

                        past_x = x
                        past_y = y

                        target_masks.append(mask_image_target)

                        # cv2.circle(get_map, (target_cord[0], target_cord[1]), 2, (0, 0, 255), 2)

                        #generate mask_agents
                        for box_id in boxes_current_frame:
                            x_, y_, w_, h_, id_ = box_id
                            if track_object_id != id_:
                                disp_x = x - x_
                                disp_y = y - y_
                                agent_cord = (target_cord[0] - disp_x, target_cord[1] - disp_y)
                                agent_cord = np.clip(agent_cord, 5, 215)                    

                                # if agent_cord[0] > 0 and agent_cord[0] < 224 and agent_cord[1] > 0 and agent_cord[1] < 224:        
                                    # cv2.circle(get_map, (agent_cord[0], agent_cord[1]), 2, (255, 0, 0), 2)
                                cv2.rectangle(mask_image_agent, (agent_cord[0]-int(width_vehicle[0]/2), agent_cord[1]-int(width_vehicle[1]/2)), (agent_cord[0] + int(width_vehicle[0]/2), agent_cord[1] + int(width_vehicle[1]/2)), (255, 0, 0), -1)
                                cv2.circle(get_map1, (int(agent_cord[0]), int(agent_cord[1])), 1, (255, 0, 0), 1)

                        agent_masks.append(mask_image_agent)
                    else:
                        target_masks.append(mask_image_target)
                        agent_masks.append(mask_image_agent)

                else:
                    target_masks.append(mask_image_target)
                    agent_masks.append(mask_image_agent)

            # print(len(target_masks), len(agent_masks))

            #stacking all images - same as lyft (first all agents, then targets, then map) (from current frame to past frames)

            #Stack agent masks
            for i in range(0, len(agent_masks)):
                agents_image = np.copy(agent_masks[i])
                agent_i = np.copy(agents_image)
                agent_i = cv2.cvtColor(agent_i, cv2.COLOR_BGR2GRAY)
                # agent_i = cv2.resize(agent_i, (self.resize, self.resize))

                if i == 0:
                    current_agent_i = np.copy(agent_i)

                agent_i = np.expand_dims(agent_i, axis=2)

                if i == 0:
                    final_frame_agent = np.copy(agent_i)
                else:
                    final_frame_agent = np.concatenate((final_frame_agent, agent_i), axis=2)


            #Stack target masks
            for i in range(0, len(target_masks)):

                ego_image = np.copy(target_masks[i])
                ego_i = np.copy(ego_image)
                ego_i = cv2.cvtColor(ego_i, cv2.COLOR_BGR2GRAY)
                # ego_i = cv2.resize(ego_i, (self.resize, self.resize))

                ego_i = np.expand_dims(ego_i, axis=2)

                if i == 0:
                    final_frame_ego = np.copy(ego_i)
                else:
                    final_frame_ego = np.concatenate((final_frame_ego, ego_i), axis=2)

            #Stack stacked-agents and stacked-ego frames
            final_frame_ae = np.concatenate((final_frame_agent, final_frame_ego), axis=2)

            #Stack map image with stacked frames
            full_image = np.copy(get_map)
            full_i = np.copy(full_image)
            full_i = cv2.cvtColor(full_i, cv2.COLOR_BGR2RGB)
            # full_i = cv2.resize(full_i, (self.resize, self.resize))
            final_frame = np.concatenate((final_frame_ae, full_i), axis=2)

            # Normalising image  (in between 0 to 1) (as in Resnet-50 we use Relu activation, normalization should be in between 0 to 1)
            final_frame = final_frame / 255.0
            final_frame = np.transpose(final_frame, (2, 0, 1)) #image = np.transpose(image, (2, 0, 1))

            # print(final_frame.shape)
            image = torch.unsqueeze(torch.tensor(final_frame, dtype=torch.float), dim=0)
            # print(image.size())
            outputs, _ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            outputs = outputs.detach().cpu().numpy()
            outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
            outputs = outputs.reshape(-1,2)
            # print(outputs.shape)
            for output_ in outputs:
                if output_[0] < 224 and output_[1] < 224:
                    cv2.circle(get_map1, (int(output_[0]), int(output_[1])), 1, (0, 0, 255), 1)

            outputs = outputs - 112
            for output_ in outputs:
                if output_[0] + target_x_orig < 224 and output_[1] + target_y_orig < 224:
                    cv2.circle(roi, (int(output_[0] + target_x_orig), int(output_[1] + target_y_orig)), 1, (0, 0, 255), 1)


            # print(outputs)

            current_frame_id = current_frame_id + 1

            # cv2.putText(frame, 'Detection Region', (50, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)    
            # cv2.rectangle(frame, (0,300), (1152,420), (0, 255, 0), 1)
            # cv2.imshow("Masked Region", mask)

            im_h1 = cv2.hconcat([get_map1, roi])
            im_h2 = cv2.hconcat([target_masks[0], agent_masks[0]])
            im_v = cv2.vconcat([im_h1, im_h2])
            # video.write(im_v)

            

            # cv2.imshow("mask_image_target", target_masks[0])
            # cv2.imshow("mask_image_agent", agent_masks[0])
            # # cv2.imshow("Detection Region", roi)
            # cv2.imshow("map", get_map)
            # cv2.imshow("Detection Region", roi)
            cv2.imshow("Detection Region", im_v)
            # key = cv2.waitKey(30)
            # exit(0)
            # # cv2.imshow("Frame", frame)

            key = cv2.waitKey(30)
            if key == 27:
                break

            # if(current_frame_id == 100):
            #     video.release()        
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     exit(1)

    # video.release()        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
