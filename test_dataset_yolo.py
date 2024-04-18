import numpy as np
import os
import tensorflow as tf
import cv2
from keras.metrics import Precision, Recall
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from glob import glob
from unet import UNet
from metrics import dice_loss
import keras
from keras.config import enable_unsafe_deserialization
keras.config.enable_unsafe_deserialization()
from yolo import yolo_predict_vehicles, yolo_predict_person, yolo_predict_sign, yolo_predict_trafficlight



TEST_DIR = 'dataset/test'
image_paths = sorted(glob(os.path.join(TEST_DIR, "*.jpg")))
print(len(image_paths))

images = []
for i in image_paths:
    image = cv2.imread(i)
    images.append(image)

images = np.array(images)

print(images.shape)


import pandas as pd
from ultralytics import YOLO

def create_yolo_results_df(images, model_path):
  """
  Analyzes a list of images using YOLO and returns a DataFrame with results.

  Args:
      images: A NumPy array of shape (num_images, height, width, channels) containing images.
      model_path: Path to the YOLO model weights file.

  Returns:
      A pandas DataFrame with columns for image name, vehicle presence (list of booleans),
      person presence (boolean), sign count (integer), and traffic light count (integer).
  """

  model = YOLO(model_path)

  results = []

  for i, image in enumerate(images):

    filename = image_paths[i]

    vehicle_presence = yolo_predict_vehicles(image)
    person_present = yolo_predict_person(image)
    sign_count = yolo_predict_sign(image)
    traffic_light_count = yolo_predict_trafficlight(image)

    results.append({
        "filename": filename,
        "vehicle_presence": vehicle_presence,
        "person_present": person_present,
        "sign_count": sign_count,
        "traffic_light_count": traffic_light_count
    })

  df = pd.DataFrame(results)

  return df

model_path = "yolo/yolov8m.pt"
df = create_yolo_results_df(images, model_path)

# Print the DataFrame
print(df)
