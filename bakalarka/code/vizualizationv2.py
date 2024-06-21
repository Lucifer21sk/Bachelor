import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

# Load the model and MultiLabelBinarizer
model_path = 'model/InceptionResNetV2/test.h5' # Change this
model = tf.keras.models.load_model(model_path)
mlb_path = 'model/InceptionResNetV2/test_mlb.pkl' # Change this
mlb = joblib.load(mlb_path)

test_csv_path = 'data/test/test_split.csv' # Change this
image_base_path = 'x-rays/augmentation2/combo' # Change this

# Function to create tf.data.Dataset for testing
def create_test_dataset(csv_path, image_base_path, mlb, batch_size=16):
    df = pd.read_csv(csv_path)
    paths = df['SOPInstanceUID'].apply(lambda uid: f"{image_base_path}/{uid}.png").tolist()
    labels = mlb.transform(df['Target'].str.split().tolist())
    
    def preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [299, 299]) # Change this
        img = preprocess_input(img)
        return img, label
    
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((path_ds, label_ds))
    dataset = dataset.map(preprocess).batch(batch_size)
    return dataset

test_ds = create_test_dataset(test_csv_path, image_base_path, mlb, batch_size=16)


#---------------------------------------------------------------------------------------------------------------------#
# VISUALIZATION

# Predictions and true labels extraction
def extract_true_predictions(dataset):
    true_labels = []
    predictions = []
    for img, label in dataset:
        pred = model.predict(img)
        predictions.extend(pred)
        true_labels.extend(label.numpy())
    return np.array(true_labels), np.array(predictions)

true_labels, preds = extract_true_predictions(test_ds)
predictions_binary = (preds > 0.5).astype(int)

# Classification report
report = classification_report(true_labels, predictions_binary, target_names=mlb.classes_, output_dict=True)

# Print classification results for each class
print("Class\t\tPrecision\tRecall\t\tF1-Score")
print("="*60)

for cls in mlb.classes_:
    precision_value = report[cls]['precision']
    recall_value = report[cls]['recall']
    f1_score_value = report[cls]['f1-score']
    
    print(f"{cls:<15}\t{precision_value:.4f}\t\t{recall_value:.4f}\t\t{f1_score_value:.4f}")

# Visualization
classes = list(report.keys())[:-3]  # Exclude summary fields
precision = [report[cls]['precision'] for cls in classes]
recall = [report[cls]['recall'] for cls in classes]
f1_scores = [report[cls]['f1-score'] for cls in classes]

plt.figure(figsize=(20, 10))
x = np.arange(len(classes))
plt.bar(x - 0.2, precision, width=0.2, label='Precision', color='lightseagreen')
plt.bar(x, recall, width=0.2, label='Recall', color='mediumblue')
plt.bar(x + 0.2, f1_scores, width=0.2, label='F1 Score', color='mediumpurple')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Model Performance Metrics by Class')
plt.xticks(x, classes, rotation='vertical')
plt.legend()
plt.tight_layout()
plt.show()
