
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers , Input , Model
import cv2
import time


####################converting RGB to Grayscale
def convert_rgb_to_grayscale(root_dir):
    print("🔧 Converting RGB images to Grayscale...")
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                with Image.open(fpath) as img:
                    if img.mode == 'RGB':
                        gray = img.convert('L') 
                        gray.save(fpath)
                        print(f"✅ Converted to grayscale: {fpath}")
            except Exception as e:
                print(f"❌ Failed to convert {fpath}: {e}")

data_dir = r"D:\Edu\Social Robotics\HW1\data_train"
convert_rgb_to_grayscale(data_dir)


######################### Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.15,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest',
    validation_split=0.2  
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  
)

batch_size = 64
img_height, img_width = 48,48

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',  
    batch_size=batch_size,
    class_mode='categorical',
    subset='training', 
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',  
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation' ,
    shuffle=False 
)

def show_samples(generator):
    x_batch, y_batch = next(generator)
    for i in range(2):
        plt.imshow(x_batch[i].squeeze(), cmap='gray')
        label_index = np.argmax(y_batch[i])
        class_name = list(generator.class_indices.keys())[label_index]
        plt.title(f"Class: {class_name}\nOne-hot: {y_batch[i]}")
        plt.axis('off')
        plt.show()

show_samples(train_generator)

from collections import Counter


def count_images_per_class(generator, name=""):
    labels = generator.classes
    label_counts = Counter(labels)
    class_indices = generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    print(f"\n📊 {name} set class distribution:")
    for class_idx, count in sorted(label_counts.items()):
        class_name = idx_to_class[class_idx]
        print(f"  {class_name:20s}: {count} images")
    print(f"🔢 Total: {sum(label_counts.values())} images")


count_images_per_class(train_generator, name="Training")
count_images_per_class(val_generator, name="Validation")

####################### adjusting loss function with class weightening

def weighted_categorical_crossentropy(y_true, y_pred):
    class_weights = tf.constant( [3.5, 6.5, 2.5, 5, 1.2, 0.3, 4.7, 1.2], dtype=tf.float32)    
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    unweighted_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_loss = unweighted_loss * weights
    
    return tf.reduce_mean(weighted_loss)


######################## building model
num_classes = train_generator.num_classes
input_shape = (img_height, img_width, 1)
inputs = Input(shape=input_shape)


x = inputs 

x = layers.Conv2D(64, (3, 3), padding='same')(x)
#x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, (3, 3), padding='same')(x)
#x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(256, (3, 3), padding='same')(x)
#x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.GlobalMaxPooling2D()(x)
#x = layers.GlobalAveragePooling2D()(x)
#x = layers.Flatten()(x)

x = layers.Dense(64)(x)
#x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)
#x = layers.Dense(128)(x)
#x = layers.BatchNormalization()(x)
#x = layers.Activation('relu')(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()

######################### ploting model 
from tensorflow.keras.utils import plot_model

# Save and display the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

############################ Train the model

model.compile(optimizer = Adam(0.00008), loss=weighted_categorical_crossentropy, metrics=['accuracy'])

checkpoint_cb = callbacks.ModelCheckpoint("ckplus_best_model.keras", save_best_only=True, monitor='val_accuracy')

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint_cb]
)

# Plot accuracy & loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

model.save("D:\emotion.keras")       # adjust this as you want
model = tf.keras.models.load_model("D:emotion.keras", compile=False)  # adjust this as you want

################# ploting confusion matrix
# Get true labels and predictions
val_generator.reset()
Y_true = val_generator.classes
Y_pred_probs = model.predict(val_generator)
Y_pred = np.argmax(Y_pred_probs, axis=1)

# Class labels
class_names = list(val_generator.class_indices.keys())

# Plot confusion matrix
cm = confusion_matrix(Y_true, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(Y_true, Y_pred, target_names=class_names))

################ ploting effect of some of the feature maps on the input picture 
val_generator.reset()
sample_img, _ = val_generator[0]  
img = sample_img[50]  
img_input = np.expand_dims(img, axis=0)

layer_name = 'conv2d_5'  
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
feature_maps = intermediate_layer_model.predict(img_input)

n_filters = min(16, feature_maps.shape[-1])  # Show up to 16 filters
plt.figure(figsize=(12, 8))
for i in range(n_filters):
    plt.subplot(4, 4, i+1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')
    plt.title(f'Filter {i}')
plt.suptitle(f'Feature Maps from Layer: {layer_name}', fontsize=16)
plt.tight_layout()
plt.show()


################# opening webcam and taking picture and video for final validation


save_dir = "D:\"   # adjust this 
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Capturing images...")
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    img_path = os.path.join(save_dir, f"image_{i}.jpg")
    cv2.imwrite(img_path, frame)
    time.sleep(0.5)  

print("Images saved.")

video_path = os.path.join(save_dir, "short_video.avi")
fps = 20
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

print("Recording video...")
start_time = time.time()
while time.time() - start_time < 5: 
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

print("Video saved.")

cap.release()
out.release()


IMG_SIZE = (48, 48)
class_names = list(val_generator.class_indices.keys())

for i in range(10):
    img_path = f"D/image_{i}.jpg"    # adjust this 
    img = Image.open(img_path).resize(IMG_SIZE)

    # Convert RGB to grayscale
    if img.mode == 'RGB':
        img = img.convert('L')  

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    print(f"Image {i}: Predicted '{pred_class}' with {confidence:.2f} confidence")
