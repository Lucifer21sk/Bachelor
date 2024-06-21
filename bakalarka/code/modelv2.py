import pandas as pd
import numpy as np
import os
import joblib
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input # Change this
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


# Function to load and preprocess images
def load_and_preprocess_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [299, 299]) # Change this
    img = preprocess_input(img)
    return img, label

# Function to create a dataset from a DataFrame
def create_dataset(df, image_base_path, mlb):
    # Prepare image paths and labels
    img_paths = [os.path.join(image_base_path, f"{uid}.png") for uid in df['SOPInstanceUID']]
    labels = mlb.transform(df['Target'].str.split().tolist())

    # Create a tf.data.Dataset
    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    image_label_ds = image_label_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return image_label_ds

# Load CSVs and MultiLabelBinarizer
train_df = pd.read_csv('data/test/train_split.csv') # Change this
val_df = pd.read_csv('data/test/val_split.csv') # Change this
test_df = pd.read_csv('data/test/test_split.csv') # Change this

mlb = MultiLabelBinarizer()
mlb.fit(train_df['Target'].str.split().tolist())  # Fit mlb on training labels

# Create datasets
train_ds = create_dataset(train_df, 'x-rays/augmentation2/combo', mlb).batch(16).prefetch(tf.data.AUTOTUNE) # Change this
val_ds = create_dataset(val_df, 'x-rays/augmentation2/combo', mlb).batch(16).prefetch(tf.data.AUTOTUNE) # Change this
test_ds = create_dataset(test_df, 'x-rays/augmentation2/combo', mlb).batch(16).prefetch(tf.data.AUTOTUNE) # Change this, do i need this?

# Define the model (as before)
model = Sequential([
    InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3)), #Change the model
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(mlb.classes_), activation='sigmoid')
])

for layer in model.layers[0].layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

start_time = time.time()

# Train the model using tf.data datasets
history = model.fit(train_ds, epochs=50, validation_data=val_ds)

end_time = time.time()
training_duration = end_time - start_time

# Save the model and MultiLabelBinarizer
model.save('model/InceptionResNetV2/test.h5') # Change this
joblib.dump(mlb, 'model/InceptionResNetV2/test_mlb.pkl') # Change this




#--------------------------------------------------------------------------------------------------------------#
# VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training and Validation Accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linestyle='-', marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--', marker='x')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.legend()
ax1.grid(True)

# Training and Validation Loss
ax2.plot(history.history['loss'], label='Training Loss', linestyle='-', marker='o')
ax2.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', marker='x')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Validation Loss')
ax2.legend()
ax2.grid(True)

# Display final values of accuracy and loss
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Training Duration: {int(training_duration // 60)} minutes {training_duration % 60:.0f} seconds")

plt.tight_layout()
plt.show()
