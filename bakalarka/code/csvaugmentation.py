import os
import pandas as pd
import numpy as np

# Define paths
labels_csv_path = 'archive/train.csv'
output_base_path = 'data/test'  # Change this
augmented_images_path = 'x-rays/augmentation/combo'  # Change this

# Ensure output directories exist
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# Define output CSV paths
full_csv_path = os.path.join(output_base_path, 'test.csv')  # Change this
train_csv_path = os.path.join(output_base_path, 'train_split.csv')
val_csv_path = os.path.join(output_base_path, 'val_split.csv')
test_csv_path = os.path.join(output_base_path, 'test_split.csv')

def create_full_csv(labels_csv_path, augmented_images_path, full_csv_path):
    """
    Generate a comprehensive CSV including original and augmented images with labels.
    """
    labels_df = pd.read_csv(labels_csv_path)
    full_data = []

    # Process each image in the augmented images path
    for filename in os.listdir(augmented_images_path):
        if filename.endswith('.png'):
            parts = filename.split('_', 1)
            if len(parts) > 1:
                # This is an augmented image
                augmentation_type, original_sop_instance_uid = parts[0], parts[1].replace('.png', '')
            else:
                # This is an original image
                original_sop_instance_uid = filename.replace('.png', '')
                augmentation_type = None

            # Find the corresponding label in the original CSV
            matching_row = labels_df[labels_df['SOPInstanceUID'] == original_sop_instance_uid]
            if not matching_row.empty:
                target = matching_row['Target'].iloc[0]
                # For augmented images, prepend the augmentation type to the SOPInstanceUID in the CSV
                sop_instance_uid_csv = f"{augmentation_type}_{original_sop_instance_uid}" if augmentation_type else original_sop_instance_uid
                full_data.append({'SOPInstanceUID': sop_instance_uid_csv, 'Target': target})
            else:
                print(f"No matching label found for image: {filename}")

    # Create a DataFrame from the full dataset and save to CSV
    full_df = pd.DataFrame(full_data)
    full_df.to_csv(full_csv_path, index=False)
    print(f"Comprehensive images CSV created at {full_csv_path}")


def create_splits_and_save(full_csv_path, output_base_path, test_samples_per_class=15, val_samples_per_class=10):
    """
    Create train, validation, and test splits from the comprehensive CSV file and save them.
    """
    print("Creating splits and saving CSV files...")
    
    # Load the comprehensive CSV with all images
    full_df = pd.read_csv(full_csv_path)
    
    # Convert the 'Target' column to string type
    full_df['Target'] = full_df['Target'].astype(str)
    
    # Initialize containers for split indices
    train_indices, val_indices, test_indices = [], [], []
    
    # Extract unique labels/classes from the 'Labels' column
    all_labels = set()
    full_df['Target'].str.split().apply(all_labels.update)
    unique_labels = sorted(all_labels)
    
    # Dictionary to store the count of samples per class in each split
    samples_per_class = {'train': {}, 'val': {}, 'test': {}}
    
    # For each label, perform the split
    for label in unique_labels:
        relevant_rows = full_df[full_df['Target'].str.contains(f'\\b{label}\\b', regex=True)]
        relevant_indices = relevant_rows.index.tolist()
        np.random.shuffle(relevant_indices)
        
        # Allocate test and validation samples
        test_indices += relevant_indices[:test_samples_per_class]
        val_indices += relevant_indices[test_samples_per_class:test_samples_per_class + val_samples_per_class]
        
        # Store the count of samples for this class in each split
        samples_per_class['train'][label] = len(relevant_indices) - test_samples_per_class - val_samples_per_class
        samples_per_class['val'][label] = val_samples_per_class
        samples_per_class['test'][label] = test_samples_per_class
    
    # Deduplicate indices since an image can belong to multiple labels
    test_indices = list(set(test_indices))
    val_indices = list(set(val_indices).difference(test_indices))
    
    all_indices = set(range(len(full_df)))
    train_indices = list(all_indices.difference(test_indices).difference(val_indices))
    
    # Split the DataFrame according to the indices and save to CSV
    full_df.iloc[train_indices].to_csv(train_csv_path, index=False)
    full_df.iloc[val_indices].to_csv(val_csv_path, index=False)
    full_df.iloc[test_indices].to_csv(test_csv_path, index=False)
    
    print("Splitting and CSV file creation completed!")
    
    # Print the count of samples per class in each split
    print("\nSamples per class in each split:")
    for split, class_count in samples_per_class.items():
        print(f"{split.capitalize()} Split:")
        for label, count in class_count.items():
            print(f"Class {label}: {count} samples")

# Create the full CSV including augmented images
create_full_csv(labels_csv_path, augmented_images_path, full_csv_path)

# Now, call create_splits_and_save with the full images CSV path and output directory
create_splits_and_save(full_csv_path, output_base_path)
