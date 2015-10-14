package edu.neuralnetwork.learning;

import java.util.Arrays;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

public class NeuralNetwork {
	
	public double[][] X, y;
	public int layer1, layer2, layer3;
	public double[] theta;
	
	/**
	 * Create a new neural network.
	 * @param X List of training examples. Each row is a different example. 
	 * Each column is a position in the input layer.
	 * @param y List of classification labels for all examples
	 * @param hiddenLayer Size of hidden layer
	 * @param outputLayer Size of output layer
	 * @param epsilon Weights are randomly initialized as values within range 
	 * (-epsilon, epsilon)
	 */
	public NeuralNetwork(double[][] X, int[] y, int hiddenLayer, int outputLayer, double epsilon) {
		this.X = X;
		this.y = NeuralNetwork.convertYData(y, outputLayer);
		layer1 = X[0].length;
		layer2 = hiddenLayer;
		layer3 = outputLayer;
		theta = NeuralNetwork.randomInitializeTheta(layer1, layer2, layer3, epsilon);
	}
	
	/**
	 * Uses forward propagation through neural network to calculate predicted values.
	 * @param X List of test examples. Each row is a different example. 
	 * Each column is a position in the input layer.
	 * @return array representation of output (classification) layer
	 */
	public double[][] propagate(double[][] X, double[] theta) {
		// Set up matrices
		DoubleMatrix Xmat = new DoubleMatrix(X);
		int m = X.length;
		
		// Roll up theta
		DoubleMatrix theta1 = unrollTheta(theta, 0, layer1, layer2);
		DoubleMatrix theta2 = unrollTheta(theta, theta1.getLength(), layer2, layer3);
		
		// Forward propagation
		DoubleMatrix col1 = DoubleMatrix.ones(m, 1);
		DoubleMatrix a1 = DoubleMatrix.concatHorizontally(col1, Xmat);
		DoubleMatrix z2 = a1.mmul(theta1.transpose());
		DoubleMatrix a2 = DoubleMatrix.concatHorizontally(col1, sigmoid(z2));
		DoubleMatrix z3 = a2.mmul(theta2.transpose());
		DoubleMatrix a3 = sigmoid(z3);
		
		return a3.toArray2();
	}
	
	/**
	 * Uses forward propagation through neural network to calculate predicted values.
	 * @param X List of test examples. Each row is a different example. 
	 * Each column is a position in the input layer.
	 * @return array representation of output (classification) layer
	 */
	public double[][] propagate(double[][] X) {
		return propagate(X, theta);
	}
	/**
	 * Uses forward propagation through neural network to calculate predicted values.
	 * @return array representation of output (classification) layer
	 */
	public double[][] propagate() {
		return propagate(X);
	}
	
	/**
	 * Determines the accuracy of a testing set.
	 */
	public double percentCorrect(double[][] X, double[][] y) {
		double[][] outputLayer = propagate(X);
		int numberCorrect = 0;
		for (int i=0; i<outputLayer.length; i++) {
			int maxIndex = 0; 
			double maxVal = 0;
			for (int j=0; j<outputLayer[i].length; j++) {
				if (outputLayer[i][j] > maxVal) {
					maxVal = outputLayer[i][j];
					maxIndex = j;
				}
			}
			if (y[i][maxIndex]==1) numberCorrect++;
		}
		return numberCorrect/(double)outputLayer.length;
	}
	
	/**
	 * Determines the accuracy of a training set.
	 */
	public double percentCorrectOnTrainingSet() {
		return percentCorrect(X, y);
	}
	
	/**
	 * Calculates the cost for this neural network.
	 * @param X input data
	 * @param y output data
	 * @param t set of weights and biases in neural network (theta)
	 * @param lambda regularization parameter
	 * @return cost function
	 */
	public double cost(double[][] X, double[][] y, double[] t, double lambda) {
		double[][] outputLayer = propagate(X, t);
		
		// Compute cost without regularization
		DoubleMatrix olmat = new DoubleMatrix(outputLayer);
		DoubleMatrix ymat = new DoubleMatrix(y);
		DoubleMatrix mat1 = ymat.mul(MatrixFunctions.log(olmat));
		DoubleMatrix mat2 = oneMinus(ymat).mul(MatrixFunctions.log(oneMinus(olmat)));
		double J = -(mat1.add(mat2)).sum()/y.length;
		
		// Roll up theta
		DoubleMatrix theta1 = unrollTheta(t, 0, layer1, layer2);
		DoubleMatrix theta2 = unrollTheta(t, theta1.getLength(), layer2, layer3);
		
		// Add regularization
		theta1 = theta1.getColumns(new IntervalRange(1, theta1.getColumns()));
		theta2 = theta2.getColumns(new IntervalRange(1, theta2.getColumns()));
		theta1 = theta1.mul(theta1);
		theta2 = theta2.mul(theta2);
		double reg = lambda/(2*y.length) * (theta1.sum() + theta2.sum());
		
		return J+reg;
	}
	
	/**
	 * Calculates the cost for this neural network.
	 * @param X input data
	 * @param t set of weights and biases in neural network (theta)
	 * @param lambda regularization parameter
	 * @return cost function
	 */
	public double cost(double[][] X, double[] t, double lambda) {
		return cost(X, y, t, lambda);
	}
	
	/**
	 * Calculates the cost for this neural network.
	 * @param X input data
	 * @param lambda regularization parameter
	 * @return cost function
	 */
	public double cost(double[][] X, double lambda) {
		return cost(X, theta, lambda);
	}
	
	/**
	 * Calculates the cost for this neural network.
	 * @param lambda regularization parameter
	 * @return cost function
	 */
	public double cost(double lambda) {
		return cost(X, lambda);
	}
	
	/**
	 * Calculates the gradient of theta using backpropagation.
	 * @param lambda regularization parameter
	 */
	public double[] gradient(double lambda) {
		return gradient(theta, lambda);
	}
	
	/**
	 * Calculates the gradient of theta using backpropagation.
	 * @param theta list of values to take the gradient of
	 * @param lambda regularization parameter
	 */
	public double[] gradient(double[] theta, double lambda) {
		return gradient(X, y, theta, lambda);
	}
	
	/**
	 * Calculates the gradient of theta using backpropagation.
	 * @param X List of training examples. Each row is a different example. 
	 * Each column is a position in the input layer.
	 * @param y list of output layer examples
	 * @param theta list of values to take the gradient of
	 * @param lambda regularization parameter
	 */
	public double[] gradient(double[][] X, double[][] y, double[] theta, double lambda) {
		
		// Set up matrices
		DoubleMatrix Xmat = new DoubleMatrix(X);
		DoubleMatrix ymat = new DoubleMatrix(y);
		int m = X.length;
		
		// Roll up theta
		DoubleMatrix theta1 = unrollTheta(theta, 0, layer1, layer2);
		DoubleMatrix theta2 = unrollTheta(theta, theta1.getLength(), layer2, layer3);
		
		// Forward propagation
		DoubleMatrix col1 = DoubleMatrix.ones(m, 1);
		DoubleMatrix a1 = DoubleMatrix.concatHorizontally(col1, Xmat);
		DoubleMatrix z2 = a1.mmul(theta1.transpose());
		DoubleMatrix a2 = DoubleMatrix.concatHorizontally(col1, sigmoid(z2));
		DoubleMatrix z3 = a2.mmul(theta2.transpose());
		DoubleMatrix a3 = sigmoid(z3);
		
		// Backpropagation
		DoubleMatrix delta3 = a3.sub(ymat);
		DoubleMatrix delta2 = delta3.mmul(theta2).mul(
				DoubleMatrix.concatHorizontally(col1, sigmoidGradient(z2)));
		delta2 = delta2.getColumns(new IntervalRange(1, delta2.getColumns()));
		DoubleMatrix grad1 = DoubleMatrix.zeros(theta1.getRows(), theta1.getColumns());
		DoubleMatrix grad2 = DoubleMatrix.zeros(theta2.getRows(), theta2.getColumns());
		for (int t=0; t<m; t++) {
			 grad2 = grad2.add(delta3.getRow(t).transpose().mmul(a2.getRow(t)));
			 grad1 = grad1.add(delta2.getRow(t).transpose().mmul(a1.getRow(t)));
		}
		grad1 = grad1.div(m);
		grad2 = grad2.div(m);
		
		// Regularization
		DoubleMatrix add1 = grad1
				.getColumns(new IntervalRange(1, grad1.getColumns()))
				.mul(lambda/(double)m);
		add1 = DoubleMatrix.concatHorizontally(
				DoubleMatrix.zeros(add1.rows, 1), add1);
		DoubleMatrix add2 = grad2
				.getColumns(new IntervalRange(1, grad2.getColumns()))
				.mul(lambda/(double)m);
		add2 = DoubleMatrix.concatHorizontally(
				DoubleMatrix.zeros(add2.rows, 1), add2);
		grad1 = grad1.add(add1);
		grad2 = grad2.add(add2);
		
		// Unroll theta
		DoubleMatrix thetaGrad = DoubleMatrix.concatVertically(
				grad1.reshape(grad1.getLength(), 1), 
				grad2.reshape(grad2.getLength(), 1));
		return thetaGrad.toArray();
	}
	
	/**
	 * Estimates the gradient of theta using a slow but simple algorithm.
	 * Used for making sure backpropagation is working properly.
	 * @param vals number of values to calculate gradient for
	 * @param epsilon length of interval to take gradient
	 * @param lambda regularization parameter
	 */
	public double[] gradientCheck(int vals, double epsilon, double lambda) {
		double[] gradient = new double[theta.length];
		for (int i=0; i<vals; i++) {
			theta[i] += epsilon;
			double cost1 = cost(X, theta, lambda);
			theta[i] -= 2*epsilon;
			double cost2 = cost(X, theta, lambda);
			theta[i] += epsilon;
			gradient[i] = (cost1-cost2)/(2*epsilon);
		}
		return gradient;
	}
	
	/**
	 * Use the gradient to move theta closer to the global minimum.
	 * @param gradient
	 * @param learningRate
	 */
	public void nextTheta(double[] gradient, double learningRate) {
		DoubleMatrix thetaMat = new DoubleMatrix(theta);
		DoubleMatrix gradMat = new DoubleMatrix(gradient);
		theta = thetaMat.sub(gradMat.mul(learningRate)).toArray();
	}
	
	/**
	 * Use the gradient to move theta closer to the global minimum.
	 * @param lambda regularization parameter
	 * @param learningRate
	 */
	public void nextTheta(double lambda, double learningRate) {
		nextTheta(gradient(lambda), learningRate);
	}
	
	/**
	 * Use the training data to train a neural network. Follows the gradient a
	 * certain number of iterations.
	 * @param iterations Number of iterations to get closer to the cost minimum
	 * @param lambda regulariation parameter
	 * @param learningRate
	 */
	public void learnIterations(int iterations, double lambda, double learningRate) {
		for (int i=0; i<iterations; i++) {
			nextTheta(lambda, learningRate);
			System.out.println("Iteration " + i + ", cost="+cost(lambda));
		}
	}
	
	/**
	 * Use the training data to train a neural network. Follows the gradient
	 * until a certain cost is reached.
	 * @param stopCost cost at which to stop iterating
	 * @param lambda regulariation parameter
	 * @param learningRate
	 */
	public void learnCost(double stopCost, double lambda, double learningRate) {
		while (cost(lambda) > stopCost) {
			nextTheta(lambda, learningRate);
		}
	}
	
	/**
	 * Uses the training data to train a neural network. Keeps following the
	 * gradient until the difference between costs after each iteration becomes
	 * small enough.
	 * @param diff difference between costs at stopping point
	 * @param lambda regularization parameter
	 * @param learningRate
	 */
	public void learnCostDiff(double diff, double lambda, double learningRate) {
		double lastCost = Double.MAX_VALUE;
		double thisCost = Double.MAX_VALUE/2;
		while (lastCost-thisCost > diff) {
			nextTheta(lambda, learningRate);
			lastCost = thisCost;
			thisCost = cost(lambda);
		}
	}
	
	/**
	 * Use the training data to train a neural network using conjugate gradient
	 * descent.
	 * @param iterations maximum number of times to repeat gradient descent
	 * @param threshold stop learning when magnitude of gradient reaches this value
	 * @param lambda regularization parameter
	 * @param stepSize farthest distance to look for minimum in line search
	 * @param batchSize size of mini-batches. If negative or 0, use entire training set.
	 */
	public void learnCGD(int iterations, double threshold, double lambda, double stepSize, int batchSize) {
		
		// "dT" = gradient of theta, "t" = theta, "s" = conjugate direction
		DoubleMatrix dTOld, sOld, t, dTNew, sNew;
		// Find the initial gradient of theta
		dTOld = (new DoubleMatrix(gradient(theta, lambda))).mul(-1);
		// Let the conjugate direction be the same as the gradient at first
		sOld = dTOld;
		// Initialize theta
		t = new DoubleMatrix(theta);
		// Calculate alpha
		double alpha = getAlpha(X, y, t, dTOld, sOld, stepSize, lambda);
		// Calculate the new value of theta
		t = t.add(dTOld.mul(alpha));
		
		for (int i=1; i<=iterations; i++) {
			// Create a minibatch
			double[][] xbatch;
			double[][] ybatch;
			if (batchSize > 0) {
				int[] indices = getRandomIndices(batchSize);
				xbatch = getMiniBatch(indices, X);
				ybatch = getMiniBatch(indices, y);
			} else {
				xbatch = X;
				ybatch = y;
			}
			// Find the new gradient of theta
			dTNew = (new DoubleMatrix(gradient(xbatch, ybatch, t.toArray(), lambda))).mul(-1);
			double gradNorm = dTNew.norm2();
			// Calculate the value of beta using the Polak-Ribiere formula
			double beta = (dTNew.transpose().mmul(dTNew.sub(dTOld)))
					.div(dTOld.transpose().mmul(dTOld)).sum();
			// Calculate the new direction based on the conjugate
			sNew = dTNew.add(sOld.mul(beta));
			// Calculate alpha
			alpha = getAlpha(xbatch, ybatch, t, dTNew, sNew, stepSize, lambda);
			// Move theta in the right direction
			t = t.add(sNew.mul(alpha));
			// Make the necessary swaps
			dTOld = dTNew;
			sOld = sNew;
			// If the gradient is flat enough, break
			System.out.println(i + ": gradNorm="+gradNorm);
			if (gradNorm < threshold) break;
		}
		System.out.println("cost="+cost(lambda));
		theta = t.toArray();
	}
	
	/**
	 * Takes in a list of y classifiers and converts each into a
	 * binary array.
	 */
	public static double[][] convertYData(int[] y, int maxY) {
		double[][] y2 = new double[y.length][maxY];
		for (int i=0; i<y.length; i++) y2[i][y[i]] = 1;
		return y2;
	}
	
	/**
	 * Initialize neural network by finding random values for theta
	 * @param layer1 number of neurons in input layer
	 * @param layer2 number of neurons in hidden layer
	 * @param layer3 number of neurons in output layer
	 * @param epsilon random values will be generated within range (-epsilon, epsilon)
	 */
	public static double[] randomInitializeTheta(int layer1, int layer2, 
			int layer3, double epsilon) {
		double[] theta = new double[(layer1+1)*layer2+(layer2+1)*layer3];
		Random r = new Random();
		for (int i=0; i<theta.length; i++) {
			theta[i] = Math.random()*epsilon*2 - epsilon;
		}
		return theta;
	}
	
	private static DoubleMatrix sigmoid(DoubleMatrix dm) {
		return MatrixFunctions.pow((MatrixFunctions.exp(dm.mul(-1))).add(1), -1);
	}
	
	private static DoubleMatrix sigmoidGradient(DoubleMatrix dm) {
		return sigmoid(dm).mul(oneMinus(sigmoid(dm)));
	}
	
	private static DoubleMatrix oneMinus(DoubleMatrix dm) {
		return DoubleMatrix.ones(dm.rows, dm.columns).sub(dm);
	}
	
	private static DoubleMatrix unrollTheta(double[] theta, 
			int start, int lay1, int lay2) {
		int thetaSize = (lay1+1)*lay2;
		double[] thetaArr = Arrays.copyOfRange(theta, start, start+thetaSize);
		return new DoubleMatrix(thetaArr).reshape(lay2, lay1+1);
	}
	
	/**
	 * Dynamically calculate learning rate for conjugate gradient descent.
	 * Uses the Armijo-Goldstein condition to find alpha.
	 */
	private double getAlpha(double[][] X, double[][] y, DoubleMatrix theta, 
			DoubleMatrix thetaGrad, DoubleMatrix direction, 
			double stepSize, double lambda) {
		
		double tau=0.8, c=0.8;
		double m = direction.dot(thetaGrad);
		double t = c*m;
		double alpha = stepSize/tau;
		double cost1 = cost(X, y, theta.toArray(), lambda);
		double cost2 = Double.NaN;
		
		while (Double.isNaN(cost2) || cost2+alpha*t>cost1) {
			alpha *= tau;
			cost2 = cost(X, y, theta.add(direction.mul(alpha)).toArray(), lambda);
		}
		
		return alpha;
	}
	
	/**
	 * Finds random examples in X[][] and y[][] to use in a minibatch.
	 */
	private int[] getRandomIndices(int miniBatchSize) {
		// create list of all indices up to X.length
		int[] a = new int[X.length];
		for (int i=0; i<a.length; i++) a[i] = i;
		// randomly shuffle indices
		for (int i=a.length-1; i>0; i--) {
			int j = (int)(Math.random()*(i+1));
			int temp = a[i]; a[i] = a[j]; a[j] = temp;
		}
		// return the first part of the shuffled array
		return Arrays.copyOf(a, miniBatchSize);
	}
	
	/**
	 * Creates a minibatch from X or y data.
	 * @param indices list of indices to construct minibatch from.
	 * @param data X or y
	 */
	private double[][] getMiniBatch(int[] indices, double[][] data) {
		double[][] batch = new double[indices.length][data[0].length];
		for (int i=0; i<batch.length; i++) {
			batch[i] = data[indices[i]];
		}
		return batch;
	}
	
	/*
	private double getAlpha(DoubleMatrix theta, double cost1, 
			DoubleMatrix thetaGrad, double stepSize, double lambda) {
		
		double cost2 = cost(X, theta.add(thetaGrad.mul(stepSize)).toArray(), lambda);
		
		// if cost1 is the lowest
		if (cost2>cost1) {
			return getAlpha(theta, cost1, thetaGrad, stepSize/2, lambda);
		}
		
		// move along the line until cost2 is the lowest
		double len = stepSize*2;
		while (true) {
			double cost3 = cost(X, theta.add(thetaGrad.mul(len)).toArray(), lambda);
			if (cost2<cost3) return len-stepSize;
			cost2 = cost3;
			len += stepSize;
		}
	}
	 */
	
}
