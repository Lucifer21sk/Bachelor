import os
import cv2
import numpy as np
import pydicom
import random

# Define the base path where the original DICOM images are stored
base_path = 'archive/train'

# Define the output path where the images will be consolidated and preprocessed
output_path = 'x-rays/augmentation/noise'  # Adjust as needed

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

def augment_image(image, augmentation_type):
    """
    Apply a random noise or blur augmentation to the image.
    Returns the augmented image and the augmentation type as a string.
    """
    augmentation_type = random.choice(['gaussiannoise', 'saltpeppernoise', 'motionblur', 'gaussianblur'])

    if augmentation_type == 'gaussiannoise':
        row, col = image.shape
        mean = 0
        var = 100 # Increase this for more noise
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        augmented_image = image + gauss
        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

    elif augmentation_type == 'saltpeppernoise':
        s_vs_p = 0.5
        amount = 0.02 # Increase this for more noise
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        augmented_image = out

    elif augmentation_type == 'motionblur':
        size = 15 
        # Generate motion blur kernel
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        # Apply kernel to the input image
        augmented_image = cv2.filter2D(image, -1, kernel_motion_blur)

    elif augmentation_type == 'gaussianblur':
        # Apply Gaussian blur
        kernel_size = (5, 5)  
        sigmaX = 0
        augmented_image = cv2.GaussianBlur(image, kernel_size, sigmaX)

    return augmented_image, augmentation_type

def save_image(image, filename, output_path):
    """
    Save the image with the given filename to the specified output path.
    """
    cv2.imwrite(os.path.join(output_path, filename), image)

def preprocess_and_save_image(dicom_path, output_path, output_size=(224, 224)):
    try:
        # Load DICOM image
        dicom_image = pydicom.dcmread(dicom_path)
        image = dicom_image.pixel_array

        # Convert to uint8
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize the image to the specified output size
        image_resized = cv2.resize(image, output_size)

        # Define a unique filename using SOPInstanceUID
        unique_filename = f"{dicom_image.SOPInstanceUID}"

        # Randomly select different augmentations
        selected_augmentations = random.sample(['gaussiannoise', 'saltpeppernoise', 'motionblur', 'gaussianblur'], 4)

        # Apply each of the selected augmentations and save the augmented images
        for augmentation in selected_augmentations:
            augmented_image, augmentation_type = augment_image(image_resized, augmentation)
            augmented_filename = f"{augmentation_type}_{unique_filename}.png"
            save_image(augmented_image, augmented_filename, output_path)

    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def consolidate_dicom_images(base_path, output_path):
    for subdir, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                source_path = os.path.join(subdir, file)
                preprocess_and_save_image(source_path, output_path)

# Start the consolidation and preprocessing process
print(f"Consolidating and preprocessing DICOM images from {base_path} to {output_path}")
consolidate_dicom_images(base_path, output_path)
print("Finished consolidating and preprocessing.")
