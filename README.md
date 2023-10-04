# Leaf-Diseases-Detection-Using-Transfer-Learning
The dataset used in the above project is "PlantVillage" dataset which is open-source dataset available on Kaggle. This project is only made for potato leaf and for only two diseases i.e. early blight and late blight.
Transfer Learning for Plant Disease Detection
This README.md file provides an overview and documentation for a Python script that utilizes transfer learning for plant disease detection. The script uses TensorFlow and Keras for building and training a deep learning model based on a pre-trained VGG19 architecture.

Dependencies
Ensure you have the following Python libraries installed:

os
numpy
seaborn
matplotlib
collections
sklearn (scikit-learn)
tensorflow
keras
Usage
First, make sure you have your dataset in the directory specified by data_dir. The dataset should be organized into subdirectories, each containing images of a specific class.

Set the desired image dimensions, batch size, and other hyperparameters.

Run the script, and it will perform the following steps:

Load and preprocess the dataset using tf.keras.preprocessing.image_dataset_from_directory.
Visualize the class distribution of the training and validation datasets.
Create a VGG19-based transfer learning model with custom dense layers.
Compile the model with the Adam optimizer and sparse categorical cross-entropy loss.
Train the model on the training dataset and validate it on the validation dataset, with early stopping to prevent overfitting.
Plot accuracy and loss curves for both training and validation.
Evaluate the model on the validation dataset and display the accuracy and loss.
The script also includes a function prediction(img) that allows you to input an image file path, load the model, and make predictions on the given image. It displays a bar chart showing the model's confidence in each class.

Model Summary
After training, the script displays a summary of the model's architecture, including layer names, output shapes, and the number of trainable parameters.

Confusion Matrix and Classification Report
The script calculates a confusion matrix and a classification report to evaluate the model's performance on the validation dataset. It provides precision, recall, F1-score, and support for each class, along with a macro and weighted average.

GPU Availability Check
Finally, the script checks for the availability of GPU devices if you are using a compatible environment. It lists available GPU devices using TensorFlow.

Example Usage
To demonstrate the usage of the prediction(img) function, two example image paths (Dtest.jpg and late.JPG) are provided at the end of the script. You can replace these paths with your own images to make predictions.
