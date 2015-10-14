package edu.neuralnetwork.learning;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;


public class DataLoader {
	
	/**
	 * Loads data from file and returns as one big matrix.
	 * @param filename
	 * @throws IOException
	 */
	public static double[][] getData(String filename) throws IOException {
		
		// Put the data in an ArrayList matrix
		ArrayList<ArrayList<Double>> arr = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
		String line = "";
		while ((line=reader.readLine()) != null) {
			String[] strlist = line.split(" ");
			arr.add(new ArrayList<Double>());
			for (int i=0; i<strlist.length; i++) {
				arr.get(arr.size()-1).add(Double.parseDouble(strlist[i]));
			}
		}
		reader.close();
		
		// Turn the ArrayList matrix into an array matrix
		double[][] mat = new double[arr.size()][];
		for (int i=0; i<arr.size(); i++) {
			mat[i] = new double[arr.get(i).size()];
			for (int j=0; j<arr.get(i).size(); j++) {
				mat[i][j] = arr.get(i).get(j);
			}
		}
		
		return mat;
	}
	
	/**
	 * Loads data from file and returns as one-dimensional array.
	 * @param filename
	 * @throws IOException
	 */
	public static double[] getArrayData(String filename) throws IOException {
		// Put the data in an ArrayList matrix
		ArrayList<Double> arr = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
		String line = "";
		while ((line=reader.readLine()) != null) {
			arr.add(Double.parseDouble(line));
		}
		reader.close();
		
		// Turn the ArrayList matrix into an array matrix
		double[] b = new double[arr.size()];
		for (int i=0; i<arr.size(); i++) {
			b[i] = arr.get(i);
		}
		
		return b;
	}
	
	/**
	 * Loads data from file. Returns as X and y matrices for a neural network. 
	 * The last column in the data file becomes y. Everything else becomes X.
	 * @param filename
	 * @throws IOException
	 */
	public static XAndYData loadXAndYData(String filename) throws IOException {
		
		// Put the data into two ArrayList matrices
		ArrayList<ArrayList<Double>> Xarr = new ArrayList<>();
		ArrayList<Integer> yarr = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
		String line = "";
		while ((line=reader.readLine()) != null) {
			String[] strlist = line.split(" ");
			Xarr.add(new ArrayList<Double>());
			for (int i=0; i<strlist.length-1; i++) {
				Xarr.get(Xarr.size()-1).add(Double.parseDouble(strlist[i]));
			}
			yarr.add(Integer.parseInt(strlist[strlist.length-1]));
		}
		reader.close();
		
		// Turn ArrayLists into arrays
		double[][] X = new double[Xarr.size()][];
		int[] y = new int[yarr.size()];
		for (int i=0; i<Xarr.size(); i++) {
			X[i] = new double[Xarr.get(i).size()];
			for (int j=0; j<Xarr.get(i).size(); j++) {
				X[i][j] = Xarr.get(i).get(j);
			}
			y[i] = yarr.get(i);
		}
		
		return new XAndYData(X, y);
	}
	
	/**
	 * Loads data from the MNIST file.
	 * @param Xfile name of file that holds input data
	 * @param yfile name of file that holds labels
	 * @param exStart number of the example to start reading data from
	 * @param batchSize number of examples to read
	 * @return matrices that hold X and y data
	 * @throws IOException
	 */
	public static XAndYData loadMNISTData(String Xfile, String yfile, 
			int exStart, int batchSize) throws IOException {
		DataInputStream labels = new DataInputStream(
				new FileInputStream(new File(yfile)));
		DataInputStream images = new DataInputStream(
				new FileInputStream(new File(Xfile)));
		
		// Read the first few numbers in the files
		int magicNumber = labels.readInt();
	    if (magicNumber != 2049) {
	      System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
	      System.exit(0);
	    }
	    magicNumber = images.readInt();
	    if (magicNumber != 2051) {
	      System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
	      System.exit(0);
	    }
	    int numLabels = labels.readInt();
	    int numImages = images.readInt();
	    int numRows = images.readInt();
	    int numCols = images.readInt();
	    if (numLabels != numImages) {
	      System.err.println("Image file and label file do not contain the same number of entries.");
	      System.err.println("  Label file contains: " + numLabels);
	      System.err.println("  Image file contains: " + numImages);
	      System.exit(0);
	    }
	    
	    // Read the data
	    int imgSize = numRows*numCols;
	    int[] labelArr = new int[batchSize];
	    double[][] imageArr = new double[batchSize][imgSize];
	    labels.skip(exStart);
	    images.skip(exStart*imgSize);
	    for (int i=0; i<batchSize; i++) {
	    	labelArr[i] = labels.readByte();
	    	for (int j=0; j<imageArr[i].length; j++) {
				imageArr[i][j] = (double)images.readUnsignedByte()/256.0;
			}
	    }
	    
	    labels.close();
	    images.close();
		return new XAndYData(imageArr, labelArr, numRows, numCols);
	}
	
	/**
	 * Write data to file.
	 * @param filename name of file
	 * @param data data to record
	 * @throws IOException 
	 */
	public static void writeData(String filename, double[] data) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filename)));
		writer.write("");
		for (int i=0; i<data.length; i++) {
			writer.append(data[i] + "\n");
		}
		writer.close();
	}

	/**
	 * Holds both X and y data together so they can be returned from 1 method.
	 */
	public static class XAndYData {
		
		/** input layer to neural network */
		public double[][] X;
		/** classification labels */
		public int[] y;
		/** (optional) number of rows in each image */
		public int numRows = 1;
		/** (optional) number of columns in each image */
		public int numColumns = 1;
		
		/**
		 * @param X input layer to neural network
		 * @param y classification labels
		 */
		public XAndYData(double[][] X, int[] y) {
			this.X = X;
			this.y = y;
		}
		
		/**
		 * @param X input layer to neural network
		 * @param y classification labels
		 * @param numRows (optional) number of rows in each image
		 * @param numColumns (optional) number of columns in each image
		 */
		public XAndYData(double[][] X, int[] y, int numRows, int numColumns) {
			this.X = X;
			this.y = y;
			this.numRows = numRows;
			this.numColumns = numColumns;
		}
		
	}
}
