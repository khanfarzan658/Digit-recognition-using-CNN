Handwritten Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras and can classify digits from 0 to 9.

Table of Contents
Project Overview
Dataset
Model Architecture
Setup and Installation
Training the Model
Evaluation
Visualization
Usage
License
Project Overview
This project aims to build a neural network model capable of recognizing handwritten digits. The MNIST dataset, consisting of 70,000 images, is used to train and test the model. The model is implemented using a Convolutional Neural Network (CNN) architecture to achieve high accuracy in digit recognition.

Dataset
The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits, each of size 28x28 pixels. The dataset is preloaded in TensorFlow/Keras.

Model Architecture
The CNN model used in this project consists of the following layers:

Convolutional Layer: 32 filters, kernel size of 3x3, ReLU activation function, input shape (28, 28, 1)
MaxPooling Layer: Pool size of 2x2
Convolutional Layer: 32 filters, kernel size of 3x3, ReLU activation function
MaxPooling Layer: Pool size of 2x2
Flatten Layer: Converts the 2D matrix to a 1D vector
Dense Layer: 10 neurons, sigmoid activation function (corresponding to the 10 digit classes)
Setup and Installation
To run this project, you need to have Python and TensorFlow installed. You can install TensorFlow using the following command:

bash
Copy code
pip install tensorflow
Training the Model
The model is trained for 5 epochs using the Adam optimizer and sparse categorical crossentropy loss function. The data is normalized by dividing the pixel values by 255.

python
Copy code
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_f, Y_TRAIN, epochs=5)
Evaluation
The model's performance is evaluated on the test dataset. The accuracy and loss are reported.

python
Copy code
model.evaluate(X_TEST_f, Y_TEST)
Visualization
To visualize the test images and the model's predictions:

python
Copy code
plt.matshow(X_TEST[1])
predict = model.predict(X_TEST_f)
np.argmax(predict[1])
Usage
Load the dataset: The MNIST dataset is automatically loaded from TensorFlow/Keras.
Train the model: Use the provided script to train the model on the training dataset.
Evaluate the model: Evaluate the model's performance on the test dataset.
Visualize predictions: Visualize and interpret the model's predictions on test images.
