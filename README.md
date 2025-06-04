# Deep Learning Projects

This repository contains different deep learning projects I have worked on.  
Each folder inside this repo is a standalone project with its own README file, code, models, and results.

---

## Projects

### [System Identification and Control Using RNNs](./System%20Identification%20and%20Control%20Using%20RNNs)
Simulate a mass-spring-damper system and build a simple RNN model to learn and control it.  
Covers system identification, classical control, and neural-network-based control.

- Techniques: SimpleRNN, LSTM, supervised learning
- Libraries: TensorFlow, Keras, Control Systems, SciPy

---
### [ECG Beat Classification using Deep Learning](./ECG%Beat%Classification%using%Deep%Learning)
Classify ECG beats from the MIT-BIH Arrhythmia dataset into five categories using a deep neural network.
Includes data preprocessing, class balancing, hyperparameter tuning with Keras Tuner, and performance evaluation.

- Techniques: Fully connected neural networks, dropout, batch normalization, L2 regularization
- Libraries: TensorFlow, Keras, Keras Tuner, scikit-learn

---
## [Emotion-Recognition-from-Facial-Images](./Emotion-Recognition-from-Facial-Images)
This project implements a deep learning pipeline to recognize human emotions from facial images using convolutional neural networks (CNNs). It covers data preprocessing (including RGB to grayscale conversion), augmentation, model training with a custom weighted loss function to handle severe class imbalance, and comprehensive evaluation with confusion matrices and classification reports. The project also includes visualization of learned feature maps and real-time emotion prediction using webcam input.

- Techniques: CNN, weighted categorical crossentropy, data augmentation, feature visualization
- Libraries: TensorFlow, Keras, OpenCV, scikit-learn, matplotlib
