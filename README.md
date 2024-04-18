# Road Scene Segmentation Project

This project aims to perform image segmentation on road scene images to identify drivable sections of road and detect cars, trucks, and people using a combination of UNet and YOLOv8 models.

## Overview

In road scene segmentation, the task is to partition an image into semantic regions, where each pixel is labeled according to its category. This project focuses on two main tasks:

1. **Road Segmentation**: Identifying drivable sections of the road within a given scene. This is crucial for autonomous vehicles and various road infrastructure applications.

2. **Object Detection**: Detecting and classifying cars, trucks, and people within the scene. This task provides valuable information for traffic monitoring, safety analysis, and navigation systems.

## Models Used

### UNet

UNet is a convolutional neural network architecture used for biomedical image segmentation. It consists of a contracting path to capture context and a symmetric expanding path to enable precise localization. In this project, UNet is utilized to perform road segmentation by labeling each pixel as drivable or non-drivable.

### YOLOv8

YOLOv8 (You Only Look Once) is a state-of-the-art object detection model known for its speed and accuracy. It divides the input image into a grid and predicts bounding boxes and class probabilities for objects within each grid cell. In this project, YOLOv8 is employed to detect cars, trucks, and people within the road scene.

## Usage

1. **Data Preparation**: Organize your dataset into appropriate directories (e.g., `images/train`, `masks/train`, `images/val`, `masks/val`, etc.).

2. **tf.data.Dataset** is used to prefetch and load the data for model training.

3. **Training UNet**: Train the UNet model on the road segmentation task using the training set. Adjust hyperparameters as needed and monitor performance on the validation set.

    **Running Inference on YOLOv8**: The yolov8 model is pre-trained on the COCO dataset and is used only for inference

4. **Evaluation**: Evaluate the trained models on the test set to assess their performance. Calculate metrics such as Intersection over Union (IoU) for road segmentation and Average Precision (AP) for object detection.

5. **Inference**: Perform inference on new road scene images using the trained models to segment drivable sections of road and detect cars, trucks, and people. Visualize the results and analyze model predictions.

**Create Conda Virtual Environment**:

    ```
    conda create --name road-segmentation python=3.x
    conda activate road-segmentation
    ```

## Requirements

    ```
    pip install -r requirements.txt

    ```

