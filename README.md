# signlang-handDetection: Hand Detection using Mediapipe framework

This repository contains code for a hand detection system that uses the Mediapipe framework [1]. The primary objective of this project is to extract regions of interest (ROIs) of hands from video frames, extract their features, and use them to recognize sign language through a neural network.

## Dataset: Chicago FS Wild [2]
The "Chicago FS Wild" dataset contains 7,304 sequences, out of which only 142 are accompanied by binding boxes that contain the coordinates of hand ROIs in the frames. Although this subset of 142 sequences is insufficient for training a model for sign language recognition, it can be used to evaluate a hand tracking algorithm.

## Mediapipe
Mediapipe provides landmark modules for different body parts, including the pose, face, and hand landmark modules. We used the hand landmark module to extract 21 3D hand keypoints for each hand. However, this module does not return the bounding box required for immediate export of hand ROIs, as shown in the example image below:

[example image 1 ] 

To predict the bounding box area for each hand, we used the predicted hand landmarks. Below are examples of images from the dataset with predicted bounding boxes (blue boxes) and ground truths (green boxes) provided by the dataset:

[example image 2 ] 
[example image 3 ] 
[example image 4 ] 

## Webcam Test
To test the hand detector live using your webcam, run the following command:

```
python detect_hands.py --webcam yes
```

## Feature Extraction:
To assign an input sequence to a feature sequence, we used deep learning-based architectures based on convolutional neural networks (CNNs). The feature extraction model is based on the ResNet architecture and pre-trained using ImageNet. We kept the feature layer, average pooling layer, and a fully connected layer that extracts a 1024-dimensional vector for each input image to extract features only. To run feature extraction, execute the following command:

```
python get_features.py
```


## References:
[1] Mediapipe Github page: https://github.com/google/mediapipe

[2] Chicago FS Wild Dataset: https://sagniklp.github.io/cfs.html
