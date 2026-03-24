#  Fruit Image Classifier (TensorFlow & Keras)

This project is a Convolutional Neural Network (CNN) built with Python and TensorFlow/Keras to classify images of different fruits. It features a split architecture: one script for training and saving the model, and another script with a simple GUI to select an image and make predictions instantly.

## Features
* **Custom CNN Architecture:** Uses multiple Convolutional and MaxPooling layers to extract features from images.
* **Separated Workflows:** Train the model once and use the saved model for endless predictions without retraining.
* **Simple GUI:** Uses `tkinter` to open a native file dialog box, making it easy to select test images.
* **Performance Graphs:** Automatically plots training vs. validation accuracy using `matplotlib`.
* **GPU Support:** Configured to leverage NVIDIA GPUs for faster training (if available).

## 📁 Project Structure

```text
📦 Your-Repository-Name
 ┣ 📂 trainning/           # Your training images (organized by fruit subfolders)
 ┣ 📂 test/                # Your validation/test images (organized by fruit subfolders)
 ┣ 📜 projectfruit.py      # The script to train the model and save it
 ┣ 📜 result.py            # The script to load the model and predict a selected image
 ┣ 📜 fruit_model.keras    # The saved model (generated after running projectfruit.py)
 ┗ 📜 class_names.json     # The saved class labels (generated after running projectfruit.py)
```
🛠️ Prerequisites
Make sure you have Python installed, along with the following libraries:

**`pip install tensorflow matplotlib numpy`**

##  How to Use
1. Prepare Your Data
Ensure you have your dataset organized into trainning and test folders inside the project directory. Inside these folders, create a subfolder for each fruit (e.g., trainning/Apple, trainning/Banana, etc.).

2. Train the Model
Run the training script. This will process your images, train the CNN, display an accuracy graph, and save both the model (fruit_model.keras) and the class names (class_names.json).
python projectfruit.py

##  Model Details
The neural network is built using the Sequential API in Keras and includes:

4 Conv2D layers with ReLU activation for feature extraction.

4 MaxPooling2D layers to reduce spatial dimensions.

A Flatten layer to convert the 2D matrices into a 1D vector.

A Dense hidden layer with 512 neurons.

A Dropout layer (0.5) to prevent overfitting.

A final Dense output layer using the Softmax activation function to determine the most likely fruit class.
