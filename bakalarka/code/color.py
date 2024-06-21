import os
import cv2
import numpy as np
import pydicom
import random

# Define the base path where the original DICOM images are stored
base_path = 'archive/train'

# Define the output path where the images will be consolidated and preprocessed
output_path = 'x-rays/augmentation/color'  

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

def augment_image(image, augmentation_type):
    """
    Apply a random color augmentation to the image: Brightness Adjustment, Contrast Adjustment, 
    Histogram Equalization, or Negative Transformation.
    Returns the augmented image and the augmentation type as a string.
    """
    augmentation_type = random.choice(['brightness', 'contrast', 'histogram', 'negative'])

    if augmentation_type == 'brightness':
        # Randomly choose to increase or decrease brightness
        factor = random.uniform(0.5, 1.5)  
        augmented_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

    elif augmentation_type == 'contrast':
        # Randomly choose to increase or decrease contrast
        factor = random.uniform(0.5, 1.5)  
        augmented_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

    elif augmentation_type == 'histogram':
        # Apply histogram equalization
        if len(image.shape) == 2:  # Grayscale image
            augmented_image = cv2.equalizeHist(image)
        else:  # Color image, apply equalization to each channel
            channels = cv2.split(image)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            augmented_image = cv2.merge(eq_channels)

    elif augmentation_type == 'negative':
        # Create negative image
        augmented_image = 255 - image

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
        selected_augmentations = random.sample(['brightness', 'contrast', 'histogram', 'negative'], 4)

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
