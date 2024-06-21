import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input # Change this


# Paths to CSV files
train_csv_path = 'data/test/DenseNet121/train_split.csv' # change this
val_csv_path = 'data/test/DenseNet121/val_split.csv' # change this
test_csv_path = 'data/test/DenseNet121/test_split.csv' # change this

# Base path to the directory containing all of images
image_base_path = 'x-rays/test/DenseNet121' # change this

# Function to load the dataset
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

# Load the saved model 
model = load_model('model/DenseNet/test.h5') # Change this
# Load MultiLabelBinarizer
mlb = joblib.load('model/DenseNet/test_mlb.pkl') # Change this 

# Load test dataset
test_images, test_labels, _ = load_dataset(test_csv_path, image_base_path, mlb) 



#---------------------------------------------------------------------------------------------------------------------#
# Evaluate the model on the test set
predictions = model.predict(test_images)
predictions_binary = np.where(predictions > 0.5, 1, 0)

# Generate the classification report
target_names = [str(cls) for cls in mlb.classes_]
report = classification_report(test_labels, predictions_binary, target_names=target_names, output_dict=True, zero_division=0)

# Print classification results for each class
print("Class\t\tPrecision\tRecall\t\tF1-Score")
print("="*60)

for cls in target_names:
    precision_value = report[cls]['precision']
    recall_value = report[cls]['recall']
    f1_score_value = report[cls]['f1-score']
    
    print(f"{cls:<15}\t{precision_value:.4f}\t\t{recall_value:.4f}\t\t{f1_score_value:.4f}")


# Extracting values for plotting
classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'
precision = [report[cls]['precision'] for cls in classes]
recall = [report[cls]['recall'] for cls in classes]
f1_score = [report[cls]['f1-score'] for cls in classes]

# Plotting
plt.figure(figsize=(14, 6))
index = np.arange(len(classes))
bar_width = 0.25

plt.bar(index, precision, bar_width, label='Precision', color='lightseagreen')
plt.bar(index + bar_width, recall, bar_width, label='Recall', color='mediumblue')
plt.bar(index + 2 * bar_width, f1_score, bar_width, label='F1-Score', color='mediumpurple')

plt.xlabel('Class')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.xticks(index + bar_width, classes, rotation=90)

plt.legend()

plt.tight_layout()
plt.show()
