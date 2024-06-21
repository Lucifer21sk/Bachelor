import os
import pandas as pd
import numpy as np
from collections import Counter

# Define paths
original_csv_path = 'data/basic/basic.csv'
output_base_path = 'data/filter'
filtered_csv_path = os.path.join(output_base_path, 'filter.csv')

# Ensure the output directory exists
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path, exist_ok=True)

def filter_classes_and_save(input_csv_path, output_csv_path, threshold=50):
    labels_df = pd.read_csv(input_csv_path)
    # Convert 'Target' column to list of labels
    labels_df['LabelsList'] = labels_df['Target'].astype(str).str.split()
    # Flatten list and count occurrences of each label
    all_labels = [label for sublist in labels_df['LabelsList'] for label in sublist]
    label_counts = Counter(all_labels)
    # Identify labels to drop
    labels_to_drop = {label for label, count in label_counts.items() if count < threshold}
    
    print("Dropped classes:")
    for label in labels_to_drop:
        print(f"- {label}")
    
    # Filter labels in 'LabelsList'
    labels_df['FilteredLabels'] = labels_df['LabelsList'].apply(lambda labels: ' '.join([label for label in labels if label not in labels_to_drop]))
    # Create new DataFrame for saving
    filtered_df = labels_df[['SOPInstanceUID', 'FilteredLabels']].rename(columns={'FilteredLabels': 'Target'})
    # Optionally, remove rows with no labels left after filtering
    filtered_df = filtered_df[filtered_df['Target'] != '']
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"Filtered dataset saved to {output_csv_path}.")

def create_splits_and_save(labels_csv_path, output_base_path, test_samples_per_class=5, val_samples_per_class=2):
    # Load the filtered labels CSV
    labels_df = pd.read_csv(labels_csv_path)
    
    # Initialize containers for split indices
    train_indices, val_indices, test_indices = [], [], []
    
    # Extract unique labels/classes from the 'Target' column
    all_labels = set()
    labels_df['Target'].str.split().apply(all_labels.update)
    unique_labels = sorted(all_labels)
    
    # Dictionary to store the count of samples per class in each split
    samples_per_class = {'train': {}, 'val': {}, 'test': {}}
    
    # For each label, perform the split
    for label in unique_labels:
        relevant_rows = labels_df[labels_df['Target'].str.contains(f'\\b{label}\\b', regex=True)]
        relevant_indices = relevant_rows.index.tolist()
        np.random.shuffle(relevant_indices)
        
        # Allocate test and validation samples
        test_indices.extend(relevant_indices[:test_samples_per_class])
        val_indices.extend(relevant_indices[test_samples_per_class:test_samples_per_class + val_samples_per_class])
        
        # Store the count of samples for this class in each split
        samples_per_class['train'][label] = len(relevant_indices) - test_samples_per_class - val_samples_per_class
        samples_per_class['val'][label] = val_samples_per_class
        samples_per_class['test'][label] = test_samples_per_class
    
    # Deduplicate indices since an image can belong to multiple labels
    test_indices = list(set(test_indices))
    val_indices = list(set(val_indices).difference(test_indices))
    all_indices = set(range(len(labels_df)))
    train_indices = list(all_indices.difference(test_indices).difference(val_indices))
    
    # Split the DataFrame according to the indices
    train_df = labels_df.iloc[train_indices][['SOPInstanceUID', 'Target']]
    val_df = labels_df.iloc[val_indices][['SOPInstanceUID', 'Target']]
    test_df = labels_df.iloc[test_indices][['SOPInstanceUID', 'Target']]
    
    # Save to CSV
    train_df.to_csv(os.path.join(output_base_path, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_base_path, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_base_path, 'test_split.csv'), index=False)
    
    print("Splitting and CSV file creation completed.")
    
    # Print the count of samples for each class in each split
    print("\nSamples per class in each split:")
    for split, class_count in samples_per_class.items():
        print(f"{split.capitalize()} Split:")
        for label, count in class_count.items():
            print(f"Class {label}: {count} samples")

# Filter the dataset
filter_classes_and_save(original_csv_path, filtered_csv_path)

# Now proceed with splitting the filtered dataset
create_splits_and_save(filtered_csv_path, output_base_path)
