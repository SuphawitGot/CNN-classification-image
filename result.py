# result.py
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# 1. Load the Saved Model
try:
    model = tf.keras.models.load_model('fruit_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print("Could not load model. Please run projectfruit.py to train and save the model first.")
    exit()

# 2. Load the Class Names
try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except Exception as e:
    print("Could not load class names. Ensure class_names.json exists.")
    exit()

# 3. GUI for Image Selection
Tk().withdraw()  # Hide main window
file_path = askopenfilename(
    title='Select an image file',
    filetypes=[('Image Files', '*.png *.jpg *.jpeg')]
)

# 4. Prediction Logic
if file_path:
    # Load and prep image
    img = image.load_img(file_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_fruit = class_names[predicted_class_index]
    
    print(f"Predicted fruit: {predicted_fruit}")
    print(f"Confidence: {np.max(prediction) * 100:.2f}%")
else:
    print("No image selected.")