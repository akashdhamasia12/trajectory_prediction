import cv2
from tracker import *
import os
import numpy as np


map_c = cv2.imread("353082_map.png")
print(map_c.shape)

im_h1 = cv2.hconcat([map_c, map_c])
im_h2 = cv2.hconcat([im_h1, map_c])
print(im_h2.shape)

mask_image_ = np.zeros((224,im_h2.shape[1],3), np.uint8)
mask_image_.fill(255)
print(mask_image_.shape)

im_v = cv2.vconcat([im_h2, mask_image_])
print(im_v.shape)

complete_map = cv2.vconcat([mask_image_, im_v])
print(complete_map.shape)

cv2.imwrite("complete_map.png", complete_map)


# im_h2 = cv2.hconcat([target_masks[0], agent_masks[0]])
# im_v = cv2.vconcat([im_h1, im_h2])


# dim = (224*3, 224*3)
# complete_map = cv2.resize(complete_map, dim, interpolation =cv2.INTER_AREA)
# print(complete_map.shape)
