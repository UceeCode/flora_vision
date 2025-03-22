import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

dataset, info = tfds.load("oxford_flowers102", as_supervised=True, with_info=True)

train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

def preprocess(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label

train_data = train_data.map(preprocess).batch(32)
val_data = val_data.map(preprocess).batch(32)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(102, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)

model.save('flora_vision_model.h5')
print('Model trained and saved')
