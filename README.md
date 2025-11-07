# MNIST-Digit-Classifier
This repository is used for documenting the implementation of a linear classifier for the MNIST handwritten digit dataset. 

To do this, I downloaded the MNIST dataset from Kaggle here: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

I then used a standard linear classifier structure across the 28x28 pixel image that MNIST datasets use and trained it on the dataset, leaving 10% out to be used as verification.

I then drew some digits in MS paint that are 28x28 pixels and placed them in the "my handdrawn digits" folder for the model to evaluate. 

## To Run on Your Device
As long as you have Python installed it should run fine from download, just run main.py in a terminal.

If everything worked, you should see it training for 6 epochs and then testing on any imagees in "my handdrawn digits".

To add your own digits for testing, you can use MS paint or any drawing application (such as GIMP or Photoshop), just make sure each image is 28x28 pixels. Then place your images in the "my handdrawn digits" folder and run main.py again. You should see your images being used after training.

Example output: <img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/94c24b7b-22df-479b-9e29-c591d88fa269" />


## Libraries Used

- Python `os` — file management
- OpenCV (`cv2`) — load and manipulate images
- NumPy — numerical operations
- Matplotlib — display images
- TensorFlow / Keras — build, train, save, and load the neural network
