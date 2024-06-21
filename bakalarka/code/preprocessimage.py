import os
import shutil
import cv2
import numpy as np
import pydicom

# Define the base path where the original DICOM images are stored
base_path = 'archive/train'

# Define the output path where the images will be consolidated and preprocessed
output_path = 'x-rays/basic' #change this 

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

def preprocess_and_save_image(dicom_path, output_path, output_size=(224, 224)):
    """
    Load a DICOM image, preprocess it (resize and normalize), and save as PNG.
    """
    try:
        # Load DICOM image
        dicom_image = pydicom.dcmread(dicom_path)
        image = dicom_image.pixel_array

        # Preprocess: Resize and Normalize
        image_resized = cv2.resize(image, output_size)
        image_normalized = cv2.normalize(image_resized, None, 0, 255, cv2.NORM_MINMAX)

        # Convert image to uint8 (if not already)
        image_uint8 = image_normalized.astype(np.uint8)

        # Define unique filename using SOPInstanceUID
        unique_filename = f"{dicom_image.SOPInstanceUID}.png"

        # Save preprocessed image as PNG
        output_file_path = os.path.join(output_path, unique_filename)
        cv2.imwrite(output_file_path, image_uint8)

    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def consolidate_dicom_images(base_path, output_path):
    """
    Walk through the base_path, preprocess each DICOM file found, and
    save it into the output_path directory.
    """
    for subdir, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                source_path = os.path.join(subdir, file)
                preprocess_and_save_image(source_path, output_path)

# Execute the function to start the consolidation and preprocessing process
print(f"Consolidating and preprocessing DICOM images from {base_path} to {output_path}")
consolidate_dicom_images(base_path, output_path)
print("Finished consolidating and preprocessing.")
