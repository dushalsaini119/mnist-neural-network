# Handwritten Digit Classification using Neural Networks (MNIST)

## Overview
This project implements an end-to-end machine learning pipeline to classify handwritten digits using the MNIST dataset. A neural network model was built and trained using TensorFlow, achieving ~97.4% accuracy on the test dataset.

## Dataset
- Source: TensorFlow Datasets (MNIST)
- Images: 28×28 grayscale
- Classes: Digits 0–9
- The dataset provides 70,000 images (28x28 pixels) of handwritten digits (1 digit per image)

## Problem Statement

The goal is to write an algorithm that detects **which digit is written** in each image.

- Since there are only **10 possible digits** (0–9), this is a **multi-class classification problem**.
- **Number of classes:** 10

## Objective

Our objective is to build a **deep neural network** with:
- **3 hidden layers**
- An output layer capable of classifying the input into one of the **10 digit classes**

## Tools & Technologies
- Python
- TensorFlow / Keras
- TensorFlow Datasets (tfds)
- NumPy

## Workflow
1. Loaded MNIST dataset using `tfds` with supervised learning setup
2. Preprocessed data by normalizing pixel values (0–1 range)
3. Created training, validation, and test datasets
4. Applied shuffling and mini-batch gradient descent
5. Built a multi-layer neural network using ReLU and Softmax activations
6. Trained the model and evaluated performance on validation and test data

## Model Architecture
- Flatten layer (28×28 → 784)
- 3 Hidden Dense layers (100 neurons, ReLU)
- Output Dense layer (10 neurons, Softmax)

## Results
- Test Accuracy: ~97.4%
- Validation accuracy closely matched test accuracy, indicating minimal overfitting
