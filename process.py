import cv2
import numpy as np
import os

def process_images(input_dir, output_dir):


    file_list = os.listdir(input_dir)

    for filename in file_list:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is not None:
            processed_image = target_color_to_white(image)

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, processed_image)

            print(f"Processed: {filename}")
        else:
            print(f"Unable to read: {filename}")


def target_color_to_white(image):
    

    target_color = [128,64,128]

    lower_bound = np.array(target_color, dtype=np.uint8)
    upper_bound = np.array(target_color, dtype=np.uint8)

    mask = cv2.inRange(image, lower_bound, upper_bound)

    image[mask == 255] = [255, 255, 255]

    image[mask != 255] = [0, 0, 0]

    return image


input_dir = "dataset/segments/"
output_dir = "dataset/segments/"


process_images(input_dir, output_dir)
