from ultralytics import YOLO
import cv2

def split_image_thirds(image):

  height, width, _ = image.shape
  part_width = width // 3
  return [image[:, :part_width], image[:, part_width:part_width * 2], image[:, part_width * 2:]]


def yolo_predict_vehicles(image):


  model = YOLO('yolo/yolov8m.pt')

  targets = [2, 5, 7]  #vehicle class ids

  results = model(image)

  parts = split_image_thirds(image)

  presence_list = [False] * 3

  for i, part in enumerate(parts):
    classes = results[0].boxes.cls[results[0].boxes.xyxy[:, 0] < (i + 1) * part.shape[1]].cpu().numpy()

    presence_list[i] = any(cls in targets for cls in classes)

  return presence_list


def yolo_predict_person(image):


  model = YOLO('yolo/yolov8m.pt')

  # Target class ID for person
  target = 0

  # Perform object detection
  results = model(image)

  # Split the image into thirds
  parts = split_image_thirds(image)

  # List to store person presence flags for each part
  presence_list = [False] * 3

  for i, part in enumerate(parts):
    # Convert class labels to NumPy array and filter for detections in the current part
    classes = results[0].boxes.cls[results[0].boxes.xyxy[:, 0] < (i + 1) * part.shape[1]].cpu().numpy()

    # Check if any person detections exist for the current part
    presence_list[i] = any(cls == target for cls in classes)

  return presence_list


def yolo_predict_sign(image):

  model = YOLO('yolo/yolov8m.pt')

  target = 11

  results = model(image)

  # Filter detections for the target class and count them
  count = sum(cls == target for cls in results[0].boxes.cls.cpu().numpy())

  # Return the count of sign detections
  return count


def yolo_predict_trafficlight(image):

  model = YOLO('yolo/yolov8m.pt')

  target = 9 

  results = model(image)

  count = sum(cls == target for cls in results[0].boxes.cls.cpu().numpy())

  return count


def yolo_predict_sign(image):

  model = YOLO('yolo/yolov8m.pt')

  target = 11

  results = model(image)

  count = sum(cls == target for cls in results[0].boxes.cls.cpu().numpy())

  return count