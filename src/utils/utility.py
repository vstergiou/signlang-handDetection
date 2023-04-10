
import os
import cv2
import random
import imageio
import numpy as np
import utils.constants as cs
import matplotlib.pyplot as plt
from utils import cv_utils, os_utils
import os.path
from os import path

def get_bbox_coords(bbox_path): 

    num_detection = 0
    xmin, ymin , xmax , ymax, label = [ np.zeros(4, dtype ='int') for _ in range(5)]
    
    if path.exists(bbox_path):
        with open(bbox_path, "r") as filestream:
            for i,line in enumerate(filestream):
                xmin[i], ymin[i] , xmax[i] , ymax[i], label[i] = [int(val) for val in line.split(',')]
                

        num_detection = i+1
   
    return  num_detection, xmin[:num_detection], ymin[:num_detection], xmax[:num_detection], ymax[:num_detection]

    

def draw_gt_bbox(image, bbox_path):

    bbox_color = (0, 255, 0)
    num_detections = 0
    ground_truths = []

    xmin, ymin , xmax , ymax, label = [ np.zeros(4, dtype ='int') for _ in range(5)]
    
    if path.exists(bbox_path):
        with open(bbox_path, "r") as filestream:
            for i,line in enumerate(filestream):
                bbox = [int(val) for val in line.split(',')]
                
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)
                ground_truths.append(bbox)

        num_detections = i+1
   
    return  num_detections, ground_truths


def draw_styled_gt_bbox(image, bbox_path):

    green_color = (0, 255, 0)
    red_color = (0, 0, 255)
    colors = [green_color, red_color]
    num_detections = 0
    ground_truths = []

    xmin, ymin , xmax , ymax, label = [ np.zeros(4, dtype ='int') for _ in range(5)]
    
    if path.exists(bbox_path):
        with open(bbox_path, "r") as filestream:
            for i,line in enumerate(filestream):
                bbox = [int(val) for val in line.split(',')]
                bbox_color = colors[bbox[4]-1]
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)
                ground_truths.append(bbox)

        num_detections = i+1
   
    return  num_detections, ground_truths

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea<0:
        interArea =0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_hand_roi(image, bbox):

    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            