# Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# Path to your training and test image folders
train_dir = 'C:\\Users\\Acer\\Documents\\GitHub\\AIFinalRealShit\\trainning'
val_dir = 'C:\\Users\\Acer\\Documents\\GitHub\\AIFinalRealShit\\test'

# Normalize pixel values (from 0–255 to 0–1)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training images
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=500,
    class_mode='categorical'
)

# Load and preprocess validation/test images
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(100, 100),
    batch_size=500,
    class_mode='categorical'
)

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Plot accuracy graphs
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Load and preprocess an image for prediction
img = image.load_img(
    'C:\\Users\\Acer\\Documents\\GitHub\\AIFinalRealShit\\test\\Apple\\r1_71.jpg',
    target_size=(100, 100)
)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Get class labels
class_names = list(train_data.class_indices.keys())
print("Predicted fruit:", class_names[predicted_class])


'''from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)'''
