import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Check if GPU is detected
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Use relative paths if the folders are in the same directory as this script
train_dir = 'trainning' 
val_dir = 'test'        

# 1. Preprocess Data
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

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(100, 100), batch_size=200, class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir, target_size=(100, 100), batch_size=200, class_mode='categorical'
)

# 2. Build the Model
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train the Model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# 4. SAVE THE MODEL AND CLASS NAMES
model.save('fruit_model.keras')
print("Model saved to 'fruit_model.keras'")

class_names = list(train_data.class_indices.keys())
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Class names saved to 'class_names.json'")

# 5. Plot Graphs
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()