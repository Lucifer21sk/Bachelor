import os
import cv2
import numpy as np
import pydicom
import random

# Define the base path where the original DICOM images are stored
base_path = 'archive/train'

# Define the output path where the images will be consolidated and preprocessed
output_path = 'x-rays/test'  # Change this

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

def augment_image(image, augmentation_type):
    """
    Apply a random augmentation to the image: slight rotation, flipping, scaling, or translation.
    """
    augmentation_type = random.choice(['rotation','flip', 'scaling', 'translation', 'brightness', 'contrast', 'histogram', 'negative',  
                                       'gaussiannoise', 'saltpeppernoise', 'motionblur', 'gaussianblur'])
    rows, cols = image.shape

    # GEOMETRIC
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
    # COLOR
    elif augmentation_type == 'brightness':
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
    #NOISE
    elif augmentation_type == 'gaussiannoise':
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
        unique_filename = f"{dicom_image.SOPInstanceUID}.png"

        # Save the original image
        save_image(image_resized, unique_filename, output_path)

        # Randomly select different augmentations
        selected_augmentations = random.sample(['rotation','flip', 'scaling', 'translation', 'brightness', 'contrast', 'histogram', 'negative',  
                                       'gaussiannoise', 'saltpeppernoise', 'motionblur', 'gaussianblur'], 12) 


        # Apply each of the selected augmentations and save the augmented images
        for augmentation in selected_augmentations:
            augmented_image, augmentation_type = augment_image(image_resized, augmentation)
            augmented_filename = f"{augmentation_type}_{unique_filename}"
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
