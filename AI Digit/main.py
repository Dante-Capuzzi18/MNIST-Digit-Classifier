import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#  -- This block of code is what TRAINS the model, the rest of the code below this is what is using this model for digit recognition --
# - data preprocessing -
# getting the dataset and splitting it into testing and training data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing the data to 0-1 bounds
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# - neural network model -
# getting a sequential model and flattening the layer into a single array instead of a 2D
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# then adding layers in between and the output layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compiling model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(x_train, y_train, epochs=6)

# saving the model
model.save('handwritten.model.keras')

# -- This block of code is for testing the neural network model --
# loading the model
model = tf.keras.models.load_model('handwritten.model.keras')

# printing out the prediction by getting each digit, inverting the color schema and predicting using our model
image_number = 1
while os.path.isfile(f"my handdrawn digits/digit {image_number}.png"):
    try:
        img = cv2.imread(f"my handdrawn digits/digit {image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
