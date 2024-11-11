# Pneumonia Classification on X-rays using Convolutional Neural Networks

## Overview

This repository contains the implementation of a **Pneumonia Classification** system on X-ray images using **Convolutional Neural Networks (CNNs)**. The project was developed as part of the **Artificial Intelligence Applications** course, under the supervision of **Prof. Petia Georgieva** (petia@ua.pt), during the **Academic Year 2021/22**.

---

## Authors

- **André Reis Fernandes**
  - MSc. Computational Engineering
  - Department of Electronics, Telecommunications and Informatics
  - Email: [andre.fernandes16@ua.pt](mailto:andre.fernandes16@ua.pt) 
  - Number: 97977

- **Gonçalo Jorge Loureiro de Freitas**
  - MSc. Computational Engineering
  - Department of Electronics, Telecommunications and Informatics
  - Email: [goncalojfreitas@ua.pt](mailto:goncalojfreitas@ua.pt)
  - Workload: 50%
  - Number: 98012

---

## Abstract

This article focuses on the research of image classification algorithms, specifically for detecting pneumonia caused by bacterial and viral infections in X-ray images. We explore various CNN architectures to optimize classification results. The algorithms developed in this project yield positive results with an F1-Score of approximately **87%**.

**Keywords**: Convolutional Neural Networks, Data Balancing, F1-Score, ROC, AUC

---

## Approach

We divide our work into **two mini-projects**, each involving different experiments and approaches to improve the performance of our model. Below is an overview of each mini-project.

---

### Mini Project 1

In this mini-project, we compare the performance of the model with both unbalanced and balanced data, fixing the number of epochs to 25. Our goal is to observe the differences between working with unbalanced and balanced data. We implement **k-fold Cross-Validation** with 10 folds to enhance the performance of the model.

The key elements for improving model performance include:
1. **Model Checkpoint**: Saves the best performing model when an epoch improves the metrics.
2. **Early Stopping**: Stops training when a monitored metric stops improving (patience = 5).

---

### Mini Project 2

For the second mini-project, we use the **balanced data approach** and compare it with three other models:
1. **Balanced Data with Data Augmentation**
2. **VGG19 Architecture**: Using pre-trained weights.
3. **InceptionV3 Architecture**: Also using pre-trained weights.

The models are evaluated using **precision**, **recall**, and **binary cross-entropy** as the loss function. A learning rate of **0.0000001** was used to ensure a smooth convergence curve.

---

## Results

The performance of each model was evaluated based on various metrics:

- **Mini Project 1**: Compared unbalanced vs. balanced datasets.
  - The results indicated that the model was not converging with 25 epochs.
  
- **Mini Project 2**: Compared balanced data, data augmentation, and pre-trained architectures.
  - **InceptionV3** showed the best results, with the model converging after 25 epochs.
  - **VGG19** showed promising results, but convergence was not reached within the 25 epochs.

---

## Model Architecture

The CNN architecture used in this project is inspired by **Abhinav Sagar's model**, which consists of:
1. Five convolutional blocks, each including:
   - Convolutional Layer
   - Max-Pooling
   - Batch Normalization
2. Flatten layer followed by four fully connected layers.
3. Dropout layers to reduce overfitting.
4. **ReLU** activation function for hidden layers and **Sigmoid** for the output layer.

---

## Data Preprocessing

- Images are resized to **180x180** to normalize and scale them appropriately for CNNs.
- **Precision** and **Recall** are used as metrics due to the unbalanced nature of the dataset.
- **Binary Cross-Entropy** is the chosen loss function.

---

## Acknowledgments

Special thanks to **Prof. Petia Georgieva** for guidance and support in the Foundations of Artificial Intelligence cours
