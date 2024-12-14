import os
import cv2
import numpy as np

def preprocess_train_images(train_dir, processed_dir, image_size, upscale_factor=1.2): # slightly upscaled the final images so that the model has more data to work with
    """
    Preprocess training images by converting to grayscale, resizing, optionally upscaling, and saving in the "pre-processed_images" directory.

    """
    # skips preprocessing if processed images already exist.
    if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
        print(f"Preprocessed images already exist in '{processed_dir}'. Skipping preprocessing.")
        images = []
        labels = []
        
        # processed images and labels
        for filename in os.listdir(processed_dir):
            filepath = os.path.join(processed_dir, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Failed to read processed image {filepath}. Skipping.")
                continue
            label = 0 if filename.lower().startswith("cat") else 1
            images.append(image)
            labels.append(label)

        images = np.array(images, dtype="float32") / 255.0
        labels = np.array(labels, dtype="int")
        return images, labels

    # preprocess images if not already done
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    images = []
    labels = []

    print(f"Processing images in '{train_dir}'")
    for filename in os.listdir(train_dir):
        filepath = os.path.join(train_dir, filename)
        if filename.lower().startswith("cat"):
            label = 0
        elif filename.lower().startswith("dog"):
            label = 1
        else:
            print(f"Skipping unrecognized file: {filename}")
            continue

        # read the image, convert to greyscale, and resize
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Failed to read image {filepath}. Skipping.")
            continue
        target_size = (int(image_size[0] * upscale_factor), int(image_size[1] * upscale_factor))
        image = cv2.resize(image, target_size)

        # Save processed image
        processed_path = os.path.join(processed_dir, filename)
        cv2.imwrite(processed_path, image)

        images.append(image)
        labels.append(label)

    if len(images) == 0:
        raise ValueError(f"No valid images were processed from the '{train_dir}' directory.")

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels, dtype="int")

    print(f"Preprocessed {len(images)} images from '{train_dir}' and saved to '{processed_dir}'.")
    return images, labels


def load_test_images(test_dir, image_size, upscale_factor=1.0):

    images = []
    ids = []

    for filename in os.listdir(test_dir):
        filepath = os.path.join(test_dir, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Failed to read image {filepath}. Skipping.")
            continue
        target_size = (int(image_size[0] * upscale_factor), int(image_size[1] * upscale_factor))
        image = cv2.resize(image, target_size)

        images.append(image)
        ids.append(int(filename.split('.')[0]))

    images = np.array(images, dtype="float32") / 255.0

    print(f"Loaded {len(images)} test images from '{test_dir}'.")
    return images, ids
