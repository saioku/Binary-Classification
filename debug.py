import cv2
import os

train_dir = "train"
for filename in os.listdir(train_dir):
    filepath = os.path.join(train_dir, filename)
    image = cv2.imread(filepath)
    if image is None:
        print(f"Invalid or corrupted image: {filepath}")
