import cv2
import numpy as np

def find_colors_in_regions(image, colors):
  """
  Checks if any of the colors are present in specific regions of an image.

  Args:
      image: RGB image as a NumPy array.
      colors: List of target colors in BGR format.

  Returns:
      A list of booleans indicating if any color was found in left, center, 
      and right regions of the image (length 3).
  """
  # Convert image to BGR format (OpenCV uses BGR)
  image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # Define region widths (adjust based on your needs)
  image_width = image.shape[1]
  left_width = image_width // 3
  center_width = left_width
  right_width = image_width - left_width - center_width

  # Initialize empty lists for color presence in each region
  color_found_left = [False] * len(colors)
  color_found_center = [False] * len(colors)
  color_found_right = [False] * len(colors)

  # Iterate through each color
  for i, color in enumerate(colors):
    # Convert color to NumPy array
    color_np = np.array(color, dtype="uint8")

    # Create masks for each region (left, center, right)
    left_mask = cv2.inRange(image_bgr[:, :left_width], color_np, color_np)
    center_mask = cv2.inRange(image_bgr[:, left_width:left_width + center_width], color_np, color_np)
    right_mask = cv2.inRange(image_bgr[:, left_width + center_width:], color_np, color_np)

    # Check if any non-zero pixels exist in each mask (any color found)
    color_found_left[i] = cv2.countNonZero(left_mask) > 0
    color_found_center[i] = cv2.countNonZero(center_mask) > 0
    color_found_right[i] = cv2.countNonZero(right_mask) > 0

  # Check if any color was found in each region (combine results)
  any_color_left = any(color_found_left)
  any_color_center = any(color_found_center)
  any_color_right = any(color_found_right)

  return [any_color_left, any_color_center, any_color_right]

def get_color_proportion(image, target_color):
  """
  Calculates the proportion of a specific color area in an RGB image.

  Args:
      image: RGB image as a NumPy array.
      target_color: Target color to calculate proportion for (BGR format).

  Returns:
      The proportion of the target color area as a float between 0 and 1.
  """
  # Convert image to BGR format (OpenCV uses BGR)
  image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # Convert target color to NumPy array
  target_color_np = np.array(target_color, dtype="uint8")

  # Create a mask where only target color pixels remain white
  mask = cv2.inRange(image_bgr, target_color_np, target_color_np)

  # Count the number of non-zero pixels (target color pixels)
  color_pixels = cv2.countNonZero(mask)

  # Calculate the total number of pixels in the image
  total_pixels = image.shape[0] * image.shape[1]

  # Calculate the proportion of target color pixels (float division)
  proportion = color_pixels / total_pixels

  return proportion*100

def count_objects(image, class_colors):
  """
  Counts the number of objects for each class in an RGB segmentation mask.

  Args:
      image_path: Path to the RGB segmentation mask image.
      class_colors: List of tuples representing RGB color values for each class.

  Returns:
      A dictionary where keys are class colors (tuples) and values are object counts.
  """

  mask = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  class_color_nps = [np.array(color, dtype="uint8") for color in class_colors]

  object_counts = {color: 0 for color in class_colors}

  for class_color_np in class_color_nps:
    class_mask = cv2.inRange(mask, class_color_np, class_color_np)

    # Identify connected components
    components, _ = cv2.connectedComponents(class_mask)

    object_count = components - 1
    object_counts[tuple(class_color_np)] = object_count  # Store count with color as key

  return object_counts

image_path = "dataset/segments/0a0eaeaf-9ad0c6dd_train_color.png"
image = cv2.imread(image_path)

class_colors = [ 
  (250,170,30), # Traffic Light
  (220,220,0) # Signboard
]

color_map = {
  (250,170,30) : "Traffic Light",
  (220,220,0) : "Signboard"
}

object_counts = count_objects(image, class_colors)

for color, count in object_counts.items():
  print(f"Number of objects for class {color_map[color]} : {count}")

target_color = (128, 64, 128) # road

proportion = get_color_proportion(image, target_color)

print(f"Proportion of road: {proportion:.2f}")

vehicles = [
  (0,0,142),
  (0,0,70),
  (0,60,100)
]

pedestrians = [
  (220,20,60),
  (255,0,0)
]

results_vehicles = find_colors_in_regions(image, vehicles)
results_pedestrians = find_colors_in_regions(image, pedestrians)

print(results_vehicles)
print(results_pedestrians)