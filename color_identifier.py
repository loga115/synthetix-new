import cv2
import os
from collections import Counter

def is_unique_color(color, seen_colors):
  """
  Checks if the color has already been encountered.
  """
  return tuple(color) not in seen_colors


def get_dominant_color(image):
  
  # Reduce image size for faster processing (adjust as needed)
  resized_image = cv2.resize(image, (100, 100))

  # Convert image to HSV for color comparison
  hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

  # Flatten the HSV image to a list of colors
  color_list = hsv_image.reshape(-1, 3)

  # Count color occurrences
  color_counts = Counter(tuple(color) for color in color_list)

  # Get the most frequent color (excluding black)
  dominant_color = color_counts.most_common(1)[0][0]
  if dominant_color == (0, 0, 0):  # Skip black as a dominant color
    dominant_color = color_counts.most_common(2)[1][0]

  return dominant_color

def main():
  # Folder containing images
  image_folder = "dataset/segments"

  # List to store seen colors
  seen_colors = set()

  for filename in os.listdir(image_folder):
    # Skip non-image files
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
      continue

    image_path = os.path.join(image_folder, filename)

    # Read image
    image = cv2.imread(image_path)

    # Get dominant color
    dominant_color = get_dominant_color(image)

    # Check if color is unique and not already processed
    if is_unique_color(dominant_color, seen_colors):
      # Add color to seen list
      seen_colors.add(tuple(dominant_color))

      # Print image name
      print(f"First instance of color found: {filename}")

      # Open the image (adjust window name as needed)
      cv2.imshow("Image", image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

  # No more images to process
  print("Finished processing all images.")


image_names = [
  "0004a4c0-d4dff0ad_train_color.png",
  "00054602-3bf57337_train_color.png",
  "00067cfb-e535423e_train_color.png",
  "00091078-59817bb0_train_color.png",
  "001b428f-059bac33_train_color.png",
  "001c2a14-c7138401_train_color.png",
  "0027eed2-815a0001_train_color.png",
  "002a3213-ab7f6730_train_color.png",
  "03f83bd2-8b868f35_train_color.png",
  "04200e90-4a4c631e_train_color.png",
  "047e732b-aa79a87d_train_color.png",
  "08ef9f76-37ebfa18_train_color.png",
  "1527bec6-44012e7b_train_color.png",
  "5e4d654e-654a1e2b_train_color.png",
  "5fcb2ae7-70bb5872_train_color.png",
  "7abfb361-ad1b41e8_train_color.png"
]


import os
import cv2

# Separate directory for saving unique colors
output_dir = "unique_colors"  # Change this as needed

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

image_dir = 'dataset/segments'

image_paths = []

for filename in image_names:
  image_path = os.path.join(image_dir, filename)  # Modify if images are elsewhere

  # Append image path to the list (optional)
  image_paths.append(image_path)

  # Generate unique output filename (replace with preferred naming scheme)
  output_filename = f"{output_dir}/{filename}"

  # Assuming you have OpenCV installed
  # If OpenCV is not available, you'll need an alternative image library
  try:
    # Read the image (replace with your preferred image library if not using OpenCV)
    image = cv2.imread(image_path)

    # Save the image to the output directory
    cv2.imwrite(output_filename, image)
  except Exception as e:
    print(f"Error processing image {filename}: {e}")

print(f"Unique color images saved to: {output_dir}")

# Optional: Print the list of image paths (if you populated it)
# print("List of image paths:")
# for path in image_paths:
#   print(path)

if __name__ == "__main__":
  main()
