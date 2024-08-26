import cv2
from tracker import *
import os
import numpy as np


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
    cap = cv2.VideoCapture("s40_cam_1.mp4")
    # complete_map = cv2.imread("complete_map_.png")
    # dim = (224*3, 224*3)
    # complete_map = cv2.resize(complete_map, dim, interpolation =cv2.INTER_AREA)
    # print(complete_map.shape)


    # Object detection from Stable camera
    object_detector = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=800, detectShadows=False)
    frame_track = []
    current_frame_id = 0
    # video=cv2.VideoWriter('video.mp4',-1,1,(224*2,224*2))

    counter = 0
    skip = 100
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
        print(height, width)
        cv2.imwrite("saved_image_providentia_1.png", frame)

        # cv2.imshow("Frame", frame)
        # cv2.imshow("map", complete_map)
        # key = cv2.waitKey(0)
        exit(0)

        # Extract Region of interest
        roi = frame.copy()#[70: ,0: 224]
    

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
                if y>60:
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

            prev_track_id = track_object_id

            current_frame_id = current_frame_id + 1

            # cv2.putText(frame, 'Detection Region', (50, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)    
            # cv2.rectangle(frame, (0,300), (1152,420), (0, 255, 0), 1)
            # cv2.imshow("Masked Region", mask)

            # im_h1 = cv2.hconcat([get_map1, roi])
            # im_h2 = cv2.hconcat([target_masks[0], agent_masks[0]])
            # im_v = cv2.vconcat([im_h1, im_h2])
            # video.write(im_v)

            

            # cv2.imshow("mask_image_target", target_masks[0])
            # cv2.imshow("mask_image_agent", agent_masks[0])
            # # cv2.imshow("Detection Region", roi)
            # cv2.imshow("map", get_map)
            cv2.imshow("Detection Region", roi)
            # cv2.imshow("Detection Region", im_v)
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
