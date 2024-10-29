# Adaboost-Object-Detection
This project leverages the AdaBoost algorithm and OpenCV for image classification of various fruits.

## Features

- **Efficient Image Classification**: Uses AdaBoost with a DecisionTree as the base estimator to provide high accuracy for fruit classification.
- **End-to-End Solution**: Complete pipeline from image preprocessing, model training, to predicting unseen images.
- **Model Persistence**: Saves trained models and encoders, allowing for easy reuse without retraining.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: OpenCV, NumPy, scikit-learn, and joblib.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/progspac-vnn/Adaboost-Object-Detection.git

2. Install Required Packages
   ```bash
   pip install -r requirements.txt

### Usage

1. **Train The Model**: Place your dataset in the train folder and execute the training script to train and save the model.

2. **Predict New Images**: Use the saved model to classify images by executing the prediction script with an image path.

### Project Structure

**train.py**: Script for training the AdaBoost classifier.
**predict.py**: Script for predicting the label of a new fruit image.
**requirements.txt**: Required libraries and dependencies.

### Results 



