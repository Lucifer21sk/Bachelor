import os
import cv2
import numpy as np
import pydicom
import random

# Define the base path where the original DICOM images are stored
base_path = 'archive/train'

# Define the output path where the images will be consolidated and preprocessed
output_path = 'x-rays/augmentation/geometric'  

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

def augment_image(image, augmentation_type):
    """
    Apply a random augmentation to the image: slight rotation, flipping, scaling, or translation.
    """
    augmentation_type = random.choice(['rotation','flip', 'scaling', 'translation'])
    rows, cols = image.shape

    if augmentation_type == 'rotation':
        # Rotate the image by a random angle between 10-40 degrees, either clockwise or counter-clockwise
        angle = random.randint(10, 40) * random.choice([-1, 1])  # Randomly choose direction
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        augmented_image = cv2.warpAffine(image, M, (cols, rows))

    elif augmentation_type == 'flip':
        # Randomly choose between horizontal (1) and vertical (0) flip
        flip_direction = random.choice([0, 1])
        augmented_image = cv2.flip(image, flip_direction)
        
    elif augmentation_type == 'scaling':
        # Scale the image (zoom in or out) random and keep the size 224x224
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
        temp_image = cv2.warpAffine(image, M, (cols, rows))
        augmented_image = cv2.resize(temp_image, (224, 224))  # Ensure the image is resized back to 224x224

    elif augmentation_type == 'translation':
        # Translate the image randomly within +/- 10% of the image dimension
        max_trans = 0.1
        tx = random.randint(-int(max_trans * cols), int(max_trans * cols))
        ty = random.randint(-int(max_trans * rows), int(max_trans * rows))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        augmented_image = cv2.warpAffine(image, M, (cols, rows))

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
        selected_augmentations = random.sample(['rotation','flip', 'scaling', 'translation'], 4)

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
