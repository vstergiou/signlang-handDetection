import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import argparse
import timeit
import pandas as pd 
import utils.constants as cs
from utils import utility
from os import listdir
from os.path import isfile, join
from utils import cv_utils 
from utils import utility
import cv2
import mediapipe as mp 
import os
from matplotlib import pyplot as plt 
import time
from statistics import mean 
from detect_hands import mediapipe_detection, draw_styled_landmarks
from detect_hands import draw_hand_bbox, get_hand_bbox



class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()
        model = models.resnet18(pretrained=True)
        if torch.cuda.is_available():
            self.net = model.cuda()
        else:
            self.net = model

        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, input):  # extract features from the average pooling layer

        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)  # [1, 512, 7, 7]
        output = self.net.avgpool(output)  # [1, 512, 1, 1]
        output = torch.flatten(output)  # flatten removes axis 1

        return output


def extractor_2d(path_to_frames, save_path):
    feature_extractor = net().to(device)
    feature_extractor.eval()  # to ensure that any Dropout layers are not active

    # For Resnet, the image must be at least 224, 224
    # transform image to (224x224) and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    for img in sorted(os.listdir(path_to_frames)):  # list the images inside every subfolder
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            img_path = os.path.join(path_to_frames, img)  # keep the path of every image
            img_name = os.path.splitext(img)[0]  # image name 0001 # 0001.jpg -> img
           
            img = cv_utils.read_image(img_path)
            if img is None:
                continue
            img = cv_utils.resize(img,cs.resize)
            
            # Make detections
            image, results = mediapipe_detection(img, holistic)
            draw_styled_landmarks(image, results)
            cv2.imshow('mediapipe detections', image)


            '''
            lh = [0, 0, 0, 0]
            #left hand bbox detection
            if results.left_hand_landmarks:
                lh_bbox = get_hand_bbox(results.left_hand_landmarks)
                lh_img = utility.get_hand_roi(image, lh_bbox)
                
                # remove .jpg extension and add .npy
                save_name = os.path.join(save_path, img_name + '_l' + ".npy")  # full path that leads to -> l_0001.npy
                lh_img = transform(lh_img)
                x1 = Variable(torch.unsqueeze(lh_img, dim=0).float(), requires_grad=False)  # [1, 3, 224, 224]
                #x = x.cuda()

                y1 = feature_extractor(x1)
                y1 = y1.cpu().data.numpy()

                np.save(save_name, y1)
            '''
                

            rh = [0, 0, 0, 0]
            #right hand bbox detection
            if results.right_hand_landmarks:
                rh_bbox = get_hand_bbox(results.right_hand_landmarks)
                rh_img = utility.get_hand_roi(image, rh_bbox)
                

                # remove .jpg extension and add .npy
                save_name = os.path.join(save_path, img_name  + ".npy")  # full path that leads to -> l_0001.npy
                rh_img = transform(rh_img)
                x2 = Variable(torch.unsqueeze(rh_img, dim=0).float(), requires_grad=False)  # [1, 3, 224, 224]
                #x2 = x2.cuda()

                y2 = feature_extractor(x2)
                y2 = y2.cpu().data.numpy()

                np.save(save_name, y2)
                


if __name__ == '__main__':

    mp_holistic = mp.solutions.holistic # Holistic model

    data_dir = cs.FRAMES_PATH
    output_dir = cs.FEATURES_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    for folders in sorted(os.listdir(cs.FRAMES_PATH)):
        if folders.startswith('.'):
                continue
        subfolder = os.path.join(cs.FRAMES_PATH, folders) 
        save_path = os.path.join(output_dir, os.path.split(cs.FRAMES_PATH)[-1], folders)
        

        #  create subfolders if they don't exist inside features folder
        if not os.path.isdir(save_path):
            os.mkdir(os.path.join(save_path))
            print("Created directory:  ", os.path.join(save_path))
        else:
            pass

        for sequences in sorted(os.listdir(subfolder)):
            if sequences.startswith('.'):
                continue
            path_to_frames = os.path.join(subfolder, sequences)
            path_to_features = os.path.join(save_path, sequences)
           
            if not os.path.isdir(path_to_features):

                os.mkdir(os.path.join(path_to_features))
                print("Created directory:  ", os.path.join(path_to_features))
            else:
                pass
            extractor_2d(path_to_frames, path_to_features)
   
    
