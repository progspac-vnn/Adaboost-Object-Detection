import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
adaboost_model = joblib.load('adaboost_fruit_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def load_and_preprocess_image(image_path, img_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = img.reshape(1, -1)  # Flatten the image for the model
        return img
    else:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

def predict(image_path):
    img = load_and_preprocess_image(image_path)
    prediction = adaboost_model.predict(img)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]


image_to_predict = "C:/Users/Vinesh/Downloads/MY_data/predict/f2.jpeg" 
predicted_fruit = predict(image_to_predict)
print(f"The predicted fruit is: {predicted_fruit}")