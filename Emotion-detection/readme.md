# Emotion Recognition from Facial Images

This project implements a complete pipeline for **facial emotion recognition** using deep learning with TensorFlow and Keras.
It includes data preprocessing, model training, evaluation, and real-time prediction using webcam input.

---

## Project Structure

- Convert RGB facial images to grayscale
- Perform real-time data augmentation using `ImageDataGenerator`
- Build and train a CNN 
- Use a **weighted loss function** to handle class imbalance
- Evaluate performance via confusion matrix and classification report
- Visualize feature maps from intermediate layers
- Capture and classify real-time webcam input

---

## Dataset Structure

The dataset directory `data_train/` should contain one subfolder per emotion class:

- `Angry/`
- `Disgust/`
- `Fear/`
- `Happy/`
- `Neutral/`
- `Sad/`
- `Surprise/`

Each subfolder should contain grayscale or RGB facial images labeled with the corresponding emotion.

Images are converted to grayscale and augmented using Keras' `ImageDataGenerator`.

---

## Model Architecture

- Input: Grayscale image (48x48x1)
- 3 Convolutional blocks (Conv2D → ReLU → BatchNorm → MaxPool)
- Global Max Pooling
- Fully Connected layer with ReLU
- Output layer with softmax (multi-class classification)

---

## Loss Function

Due to **severe class imbalance** in the dataset (some emotion classes have significantly fewer samples than others), using a standard categorical crossentropy loss would bias the model toward majority classes.

To address this, a **custom weighted categorical crossentropy** loss is implemented. This approach assigns a higher penalty to misclassifications of underrepresented classes, helping the model learn to recognize them more accurately.

The class weights are manually defined based on the observed class frequencies:

```python
class_weights = [3.5, 6.5, 2.5, 5, 1.2, 0.3, 4.7, 1.2]
```
---

## Training and Evaluation
- Optimizer: Adam with learning rate 0.00008
- Train-validation split: 80/20
- Evaluation includes:
  -  Confusion matrix
  -  Classification report
  -  Accuracy/Loss curves

---

## Feature Map Visualization
Feature maps from a selected convolutional layer are visualized to better understand what the model is learning from input images.

---
## Real-Time Webcam Inference
-Captures 10 images and a short video using your webcam. 
-Each image is:
  -Converted to grayscale
  -Resized and normalized
  -Passed through the trained model
-Outputs class prediction and confidence for each image
