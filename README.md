# Fake Profile Detection using ANN

This project focuses on detecting fake social media profiles using a Machine Learning approach. An **Artificial Neural Network (ANN)** model is built and trained to classify profiles as either **real** or **fake** based on a set of features derived from user behavior and metadata.

## Project Structure

- `FPD_ann.ipynb`: Google Colab notebook containing data preprocessing, ANN model building, training, and evaluation.

## Problem Statement

Social media platforms are flooded with fake profiles used for scams, misinformation, and other malicious activities. This project aims to build an AI-based model to classify whether a given profile is **real** or **fake**, using supervised learning.

## Key Features

- Data preprocessing and cleaning
- Feature selection
- ANN model with `Keras`
- Accuracy evaluation
- Model improvement suggestions

## Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- TensorFlow / Keras
- Google Colab

## Model Architecture

- Input Layer: Based on selected features
- Hidden Layers: 2 dense layers with ReLU activation
- Output Layer: Single neuron with sigmoid for binary classification
- Optimizer: Adam
- Loss Function: Binary Crossentropy

## Results

- Training Accuracy: ~98%
- Validation Accuracy: ~97%
- Confusion Matrix and Classification Report included

## How to Run

1. Open the `FPD_ann.ipynb` notebook in Google Colab.
2. Run all cells in order.
3. The model will be trained, tested, and evaluated within the notebook.

## Future Work

- Deploy as a web app using FastAPI or Flask
- Improve feature engineering
- Integrate with real-time profile data

