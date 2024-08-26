import torch
import numpy as np
import cv2
import albumentations
import config
#from model import FaceKeypointResNet50
from torchvision.models.resnet import resnet50

sequence_length = 50
past_trajectory = 10

# Model Creation
def build_model() -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet50(pretrained=True)

    # change input channels number to match the rasterizer's output
    #num_in_channels = 5*past_trajectory
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
    num_targets = 2 * (sequence_length - past_trajectory)
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)

#model = FaceKeypointResNet50(pretrained=False, requires_grad=False).to(config.DEVICE)

# load the model checkpoint
checkpoint = torch.load('../outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# capture the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if (cap.isOpened() == False):
    print('Error while trying to open webcam. Plese check again...')
 
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set up the save file path
save_path = f"{config.OUTPUT_PATH}/vid_keypoint_detection.mp4"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"{save_path}", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width, frame_height))

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        with torch.no_grad():
            image = frame
            image = cv2.resize(image, (224, 224))
            orig_frame = image.copy()
            orig_h, orig_w, c = orig_frame.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)
            outputs = model(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(-1, 2)
        keypoints = outputs
        for p in range(keypoints.shape[0]):
            cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                        1, (0, 0, 255), -1, cv2.LINE_AA)
        orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
        cv2.imshow('Facial Keypoint Frame', orig_frame)
        out.write(orig_frame)
        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
 
    else: 
        break

# release VideoCapture()
cap.release()
 
# close all frames and video windows
cv2.destroyAllWindows()