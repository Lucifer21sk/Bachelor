import os
import pandas as pd
import numpy as np

# Define paths
labels_csv_path = 'archive/train.csv'  
output_base_path = 'data/basic'
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path, exist_ok=True)

# Define output CSV paths
train_csv_path = os.path.join(output_base_path, 'train_split.csv')
val_csv_path = os.path.join(output_base_path, 'val_split.csv')
test_csv_path = os.path.join(output_base_path, 'test_split.csv')

def create_splits_and_save(labels_csv_path, output_base_path, test_samples_per_class=5, val_samples_per_class=2):
    print("Creating splits and saving CSV files...")
    
    # Load the labels CSV
    labels_df = pd.read_csv(labels_csv_path)
    
    # Convert the 'Target' column to string type
    labels_df['Target'] = labels_df['Target'].astype(str)
    
    # Initialize containers for split indices
    train_indices, val_indices, test_indices = [], [], []
    
    # Extract unique labels/classes from the 'Labels' column
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
        test_indices += relevant_indices[:test_samples_per_class]
        val_indices += relevant_indices[test_samples_per_class:test_samples_per_class + val_samples_per_class]
        
        # Store the count of samples for this class in each split
        samples_per_class['train'][label] = len(relevant_indices) - test_samples_per_class - val_samples_per_class
        samples_per_class['val'][label] = val_samples_per_class
        samples_per_class['test'][label] = test_samples_per_class
    
    # Deduplicate indices since an image can belong to multiple labels
    test_indices = list(set(test_indices))
    val_indices = list(set(val_indices).difference(test_indices))
    
    all_indices = set(range(len(labels_df)))
    train_indices = list(all_indices.difference(test_indices).difference(val_indices))
    
    # Split the DataFrame according to the indices and save to CSV
    labels_df.iloc[train_indices].to_csv(train_csv_path, index=False)
    labels_df.iloc[val_indices].to_csv(val_csv_path, index=False)
    labels_df.iloc[test_indices].to_csv(test_csv_path, index=False)
    
    print("Splitting and CSV file creation completed!")
    
    # Print the count of samples per class in each split
    print("\nSamples per class in each split:")
    for split, class_count in samples_per_class.items():
        print(f"{split.capitalize()} Split:")
        for label, count in class_count.items():
            print(f"Class {label}: {count} samples")

# Now, call create_splits_and_save with the labels CSV path and output directory
create_splits_and_save(labels_csv_path, output_base_path)
