import os
import numpy as np
import cv2
import scipy
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("",tf.config.list_physical_devices())

# Set the root directory and subfolder names
root_directory = 'train'
subfolders = ['b', 'E', 'k', 'n', 'p', 'q', 'r', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']

# Define the image size
image_height, image_width = 50, 50

# Create an ImageDataGenerator to load and augment the training data
data_generator = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Load the training data
train_data = data_generator.flow_from_directory(
    root_directory,
    target_size=(image_height, image_width),
    batch_size=32,
    classes=subfolders,
    subset='training'
)

# Load the validation data (optional, but recommended for evaluation during training)
validation_data = data_generator.flow_from_directory(
    root_directory,
    target_size=(image_height, image_width),
    batch_size=32,
    classes=subfolders,
    subset='validation'
)

# Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(subfolders), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=validation_data, epochs=10)

# Save the trained model (optional)
model.save('chess_piece_classifier.h5')

print('Training completed.')
