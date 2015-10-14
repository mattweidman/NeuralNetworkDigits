Author: Matthew Weidman

This is a neural network in Java used to recognize 28x28 pixel-sized handwritten digits.

It uses data from yann.lecun.com/exdb/mnist/ as training and test data.
The training data are stored in the 4 files that end in "ubyte." They are listed here:
train-images.idx3-ubyte: 60,000 training images
train-labels.idx1-ubyte: labels for each image in train-images
(so if the nth image is a picture of 5, the nth label is 5).
t10k-images.idx3-ubyte: 10,000 test images.
t10k-labels.idx1-ubyte: labels for each image in t10k-images 

This project also uses the jblas 1.2.4 library for matrix operations.

DataLoader.java holds convenience methods used to load data in MNIST and elsewhere.
It also has the ability to save a neural network if one has been made.

NeuralNetwork.java holds different methods used to create a neural network.
New neural networks can be made by providing data in the constructor.
Then, a neural network can learn using a .learn() method.
For example, if you created a neural network like the following:
	NeuralNetwork nn = new NeuralNetwork(...);
You can make it learn by using nn.learnCGD(), nn.learnIterations, nn.learnCost(), or nn.learnCostDiff().
learnCGD() is special because it uses stochastic gradient descent, which is usually faster than using a constant learning rate.
You can find out how well the neural network is doing by using nn.percentCorrect().

Earlier saves of neural networks are stored in theta...txt files.
The theta files are labeled with what percentage of the test set they got correct.
For exxample, the neural network saved in tehta92.9.txt had 92.9% accuracy on the test set.

An example of how the neural network can be used can be found in Main.java.