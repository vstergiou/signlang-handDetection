
import pandas as pd 
import utils.constants as cs
from utils import utility
from os import listdir
from os.path import isfile, join
from utils import cv_utils 
from utils import utility
import cv2
import numpy as np
from utils import cv_utils
import mediapipe as mp 
import os
from matplotlib import pyplot as plt 
import time
import argparse
from statistics import mean 




def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR

    return image, results

def draw_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                            # mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                            # mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                            # ) 
    #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def draw_hand_bbox(image, hand_landmarks):

    x_list = []
    y_list = [] 
    bbox_color = (255, 0, 0)
    offset = 18


    if hand_landmarks:
        for res in hand_landmarks.landmark:
            x_list.append(res.x)
            y_list.append(res.y)
     
        xmin = min(x_list)*cs.image_width-offset
        xmax = max(x_list)*cs.image_width+offset
        ymin = min(y_list)*cs.image_height-offset
        ymax = max(y_list)*cs.image_height+offset

        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), bbox_color , 2)

        return [int(xmin), int(ymin), int(xmax), int(ymax)]

def get_hand_bbox(hand_landmarks):

    x_list = []
    y_list = [] 
    offset = 18

    if hand_landmarks:
        for res in hand_landmarks.landmark:
            x_list.append(res.x)
            y_list.append(res.y)
     
        xmin = min(x_list)*cs.image_width-offset
        xmax = max(x_list)*cs.image_width+offset
        ymin = min(y_list)*cs.image_height-offset
        ymax = max(y_list)*cs.image_height+offset

        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--webcam', default="no", help='Path to images or image file')  # -i data/andy
    args = ap.parse_args()
    answer = args.webcam   #use webcam or not 

    #read and store data 
    hand_df = pd.read_csv(cs.HANDANNOTATIONS_CSV)
    chicago_df = pd.read_csv(cs.CHICAGOFS_CSV)
    merged_df = pd.merge(chicago_df, hand_df, on='filename')
    left_ious = []
    right_ious = []

    #create mediapipe models 
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    mp_hands = mp.solutions.hands 

    if answer == 'yes':

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()
                frame = cv_utils.resize(frame,cs.resize)

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

        
                draw_landmarks(image, results)
                
                #show feed
                cv2.imshow("OpenCV Feed", image )

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    else:

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for sequence in merged_df.filename:
                SEQUENCE_PATH = cs.FRAMES_PATH + sequence
                frames_list = [f for f in listdir(SEQUENCE_PATH) if isfile(join(SEQUENCE_PATH, f))]
                
                for frame in frames_list:

                    current_frame_path = SEQUENCE_PATH + cs.SLASH + frame
                    current_bbox_path = cs.BBOX_PATH + sequence + cs.SLASH + frame[:len(frame)-3] + cs.TXT

                    img = cv_utils.read_image(current_frame_path)
                    if img is None:
                        continue
            
                    img = cv_utils.resize(img,cs.resize)

                    # Make detections
                    image, results = mediapipe_detection(img, holistic)
                    
                    lh = [0, 0, 0, 0]
                    #left hand bbox detection
                    if results.left_hand_landmarks:
                        lh = draw_hand_bbox(image, results.left_hand_landmarks)
                    
                    rh = [0, 0, 0, 0]
                    #right hand bbox detection
                    if results.right_hand_landmarks:
                        rh = draw_hand_bbox(image, results.right_hand_landmarks)
                    
                    draw_landmarks(image, results)
                    #draw_styled_landmarks(image,results)   # or draw styled landmarks

                    num, gt = utility.draw_gt_bbox(image, current_bbox_path)
                    #show feed
                    cv2.imshow("OpenCV Feed", image )
                    cv2.waitKey(0)
                    

