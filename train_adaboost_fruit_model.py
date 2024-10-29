import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        
        # Ensure that label is a string and check if the folder exists
        if os.path.isdir(label_folder) and isinstance(label, str):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label.lower())  # Convert to lowercase for consistency
    return np.array(images), np.array(labels)

train_folder = "C:/Users/Vinesh/Downloads/MY_data/train"  # Replace with actual path
X_train, y_train = load_images_from_folder(train_folder)

# Encode the labels
label_encoder = LabelEncoder()
label_encoder.fit(np.unique(y_train))  # Fit on all unique labels found

y_train_encoded = label_encoder.transform(y_train)

# Preprocess and flatten the images
X_train = X_train.reshape(len(X_train), -1)

# Train the model
base_estimator = DecisionTreeClassifier(max_depth=1)
adaboost_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, learning_rate=1.0)  # Corrected parameter name
adaboost_model.fit(X_train, y_train_encoded)

# Save the model and label encoder
joblib.dump(adaboost_model, 'adaboost_fruit_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and label encoder saved successfully!")
