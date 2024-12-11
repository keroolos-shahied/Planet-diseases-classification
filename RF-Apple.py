import os
import cv2
import joblib
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the directory containing the data and list all categories (diseases and healthy)
data_dir = r"E:\task\Indigenous Dataset for Apple Leaf Disease Detection and Classification"
categories = os.listdir(data_dir)

# Initialize lists to store features and labels
features = []
labels = []

# Function to extract color histogram from an image
def extract_color_histogram(image_path):
    image = Image.open(image_path)
    image = image.resize((512, 512))
    image_np = np.array(image)
    hist = cv2.calcHist([image_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to extract texture features using LBP
def extract_texture_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Function to extract combined features (color histogram + texture)
def extract_combined_features(image_path):
    color_hist = extract_color_histogram(image_path)
    texture_feat = extract_texture_features(image_path)
    combined_features = np.hstack([color_hist, texture_feat])
    return combined_features

# Loop through each category and each image within the category to extract features and labels
for label, category in enumerate(categories):
    category_path = os.path.join(data_dir, category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        if os.path.isfile(image_path):
            features.append(extract_combined_features(image_path))
            labels.append(label)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

# Create an ensemble of Random Forests with predefined hyperparameters
rf1 = RandomForestClassifier(n_estimators=200, max_depth=60, criterion="entropy", min_samples_split=2, random_state=1, n_jobs=-1)
rf2 = RandomForestClassifier(n_estimators=400, max_depth=30, criterion="gini", min_samples_split=5, random_state=1, n_jobs=-1)
rf3 = RandomForestClassifier(n_estimators=800, max_depth=20, criterion="entropy", min_samples_split=10, random_state=1, n_jobs=-1)

ensemble_model = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)], voting='soft')
ensemble_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = ensemble_model.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after using ensemble: {accuracy * 100:.2f}%")

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=categories))

# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save the trained ensemble model to a file
model_filename = 'random_forest_Apple_diseases_tuned.pkl.pkl'
joblib.dump(ensemble_model, model_filename)
print(f" model saved as {model_filename}")



