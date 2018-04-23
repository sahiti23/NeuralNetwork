# NeuralNetwork

Data: Used Images data of bees. 

Dataset: single bee train images, single bee test images, no bee train images, no bee test images.

Model:
Created a function that is a model for recognizing honeybees based on looking at every pixel in the image. The model that I used to build the neural network is softmax regression.
Used tensorflow to train the model to recognize bees. 
Check model’s accuracy with the test data.
Used honeybee data to train a convolution Neural Network for image recognition in tensorflow.

Trained and evaluated:
Initially trained the model for 10000 iterations. Tested the model against the validation set to get an idea of accuracy. 
Model is saved and restored using session.

Program Execution:
•	Please change the path of root directory of data in ROOT_DIR.
•	Save_path in saver.save() function contains the path where the model is saved. Change that path to save it locally.
•	From that saved path, restore the model in saver.restore() function.
