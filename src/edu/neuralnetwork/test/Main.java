package edu.neuralnetwork.test;

import java.io.IOException;

import org.jblas.DoubleMatrix;

import edu.neuralnetwork.learning.DataLoader;
import edu.neuralnetwork.learning.NeuralNetwork;

public class Main {

	public static void main(String[] args) {

		// Paramaters
		int hiddenLayer = 25;
		int features = 10;
		int trainingExamples = 60000;
		int testExamples = 1000;
		double lambda = 1;
		double learningRate = .1;
		double stepSize = 3; 
		int iterations = 1000;
		double epsilon = 0.12;
		double threshold = 0.1;
		int batchSize = 1000;
		String trainImages = "train-images.idx3-ubyte";
		String trainLabels = "train-labels.idx1-ubyte";
		String testImages = "t10k-images.idx3-ubyte";
		String testLabels = "t10k-labels.idx1-ubyte";
		
		// Load training set
		System.out.println("Loading training set...");
		DataLoader.XAndYData xandy = null;
		try {
			xandy = DataLoader.loadMNISTData(trainImages, trainLabels, 0, trainingExamples);
		} catch (IOException e1) {
			e1.printStackTrace();
			System.exit(0);
		}
		double[][] X = xandy.X;
		int[] y = xandy.y;
		
		// Display one example in X
		/* int ind = 3;
		for (int i=0; i<X[ind].length; i++) {
			if (i%28==0) System.out.println();
			System.out.print(X[ind][i]+" ");
		} */
		
		// Apply super contrast to X by turning anything not white into black
		for (int i=0; i<X.length; i++) {
			for (int j=0; j<X[i].length; j++) {
				if (X[i][j] > 0.001) X[i][j] = 1.0;
			}
		}
		
		
		// Initialize neural network
		NeuralNetwork nn = new NeuralNetwork(X, y, hiddenLayer, features, epsilon);
		
		// Upload previous theta from file
		/* try {
			nn.theta = DataLoader.getArrayData("theta90.01.txt");
		} catch (IOException e1) {
			e1.printStackTrace();
		} */
		
		// Gradient check
		/* double[] gradientBP = nn.gradient(lambda);
		int vals = 100;
		double[] gradientNA = nn.gradientCheck(vals, 0.001, lambda);
		for (int i=0; i<vals; i++) {
			System.out.println(gradientBP[i] + " / " + gradientNA[i]+" = "+(gradientBP[i]/gradientNA[i]));
		} */
		
		// Find initial percent correct
		double percentCorrect = nn.percentCorrectOnTrainingSet();
		System.out.println("Percent accuracy before training: " + percentCorrect*100);
		
		// Train neural network by finding cost and gradient
		System.out.println("Learning...");
		long time1 = System.currentTimeMillis();
		nn.learnCGD(iterations, threshold, lambda, stepSize, batchSize);
		long time2 = System.currentTimeMillis();
		System.out.println("Time: " + (time2-time1) + " ms");
		
		// Find percent correct in training set
		percentCorrect = nn.percentCorrectOnTrainingSet();
		System.out.println("Percent accuracy on training set: " + percentCorrect*100);
		
		// Load test set
		System.out.println("Loading test set...");
		DataLoader.XAndYData xandytest = null;
		try {
			xandytest = DataLoader.loadMNISTData(testImages, testLabels, 0, testExamples);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
		double[][] Xtest = xandytest.X;
		int[] ypreTest = xandytest.y;
		double[][] yTest = NeuralNetwork.convertYData(ypreTest, features);
		
		// Find percent correct in test set
		percentCorrect = nn.percentCorrect(Xtest, yTest);
		System.out.println("Percent accuracy on test set: " + percentCorrect*100);
		
		// Record theta
		try {
			DataLoader.writeData("theta.txt", nn.theta);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
