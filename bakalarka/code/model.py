import pandas as pd
import numpy as np
import os
import joblib
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input # Change this
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


# Paths to CSV files
train_csv_path = 'data/test/DenseNet121/train_split.csv' # Change this
val_csv_path = 'data/test/DenseNet121/val_split.csv' # Change this
test_csv_path = 'data/test/DenseNet121/test_split.csv' # Change this

# Base path to the directory containing all of images
image_base_path = 'x-rays/test/DenseNet121' # Change this

def load_dataset(csv_path, image_base_path, mlb=None):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_base_path, row['SOPInstanceUID'] + '.png')
        img = image.load_img(img_path, target_size=(224, 224)) # Change this
        img = image.img_to_array(img)
        img = preprocess_input(img)

        images.append(img)
        label_list = list(map(int, row['Target'].strip().split()))
        labels.append(label_list)

    images = np.array(images)
    
    if mlb is None:
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
    else:
        labels = mlb.transform(labels)

    return images, labels, mlb

# Load the datasets
mlb = MultiLabelBinarizer()
train_images, train_labels, mlb = load_dataset(train_csv_path, image_base_path)
val_images, val_labels, _ = load_dataset(val_csv_path, image_base_path, mlb)
test_images, test_labels, _ = load_dataset(test_csv_path, image_base_path, mlb)

# Define the model
num_classes = len(mlb.classes_)
model = Sequential([
    DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), #Change the model
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='sigmoid')
])

# Freeze the model layers
for layer in model.layers[0].layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

start_time = time.time()

# Train the model
history = model.fit(
    train_images, train_labels,
    batch_size=16,
    epochs=50,
    validation_data=(val_images, val_labels)
)
end_time = time.time()

# Calculate the training duration
training_duration = end_time - start_time

# Save the model
model.save('model/DenseNet/test.h5') # Change this
joblib.dump(mlb, 'model/DenseNet/test_mlb.pkl') # Change this 



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
