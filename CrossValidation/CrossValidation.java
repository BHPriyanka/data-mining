import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import java.util.List;
import java.util.ArrayList;
import java.io.FileInputStream;
import java.util.Arrays;

import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import java.util.Iterator;
import java.util.Random;

import Jama.Matrix;

public class CrossValidation {
	
	public static int k =5;//lambda =1;;
	
	public static void main(String[] args) throws Exception {
		File train_excel = new File ("C:/Users/BHPriyanka/Desktop/Data Mining/Priyanka_BH_CS6220_HW1/HW1_Data/train-100-10.xlsx");
		FileInputStream fis_train = new FileInputStream(train_excel);
		XSSFWorkbook wb_train = new XSSFWorkbook(fis_train);
		XSSFSheet sheet = wb_train.getSheetAt(0);
     
		Matrix X = new Matrix(L2LinearRegression.getMatrix(sheet));
	    Matrix Y = X.getMatrix(1, X.getRowDimension() - 1, X.getColumnDimension() - 1, X.getColumnDimension() - 1);
	    Matrix modified_X = L2LinearRegression.modify_matrix(X);
		
	    //get the indices of the folds for k=5
		int[][] folds = RandomlyFoldIndices(X.getRowDimension(), k);
		Map<Integer,Map<String,double[]>> result = new HashMap<Integer,Map<String,double[]>>();
		HashMap<String,List<Integer>> test_train_indices = new HashMap();
		for(int fold=0;fold<folds.length;fold++){
			test_train_indices = get_training_test_indices(folds,modified_X.getRowDimension(), fold);
			result.put(fold, train_data_against_test(test_train_indices, modified_X, Y));
		}
		
		for(int fold=0;fold<folds.length;fold++){
			System.out.println(calculateAverage(result.get(fold).get("MSE for train data")) +" " + calculateAverage(result.get(fold).get("MSE for validation data")));
		}
							
	}
	
	private static double calculateAverage(double[] values) {
	   double sum=0.0;
	   for(int i=0;i<values.length;i++){
		   sum=sum+values[i];
	   }
		return sum/values.length;
	}
	
	public static int[][] RandomlyFoldIndices(int length, int k)
	{
		int[] inds = new int[length];

	    Random rand = new Random();
	    // initialize indices
	    for (int i = 0; i < length; i++)
	    {
	        inds[i] = i;
	    }

	    // now shuffle indices for 2 times
	    for (int st = 0; st < 2; st++)
	    {
	        for (int i = 0; i < length - 1; i++)
	        {
	            // r is in [i + 1, length)
	            int r = rand.nextInt(length - i - 1)+1;
	            int temp = inds[i];
	            inds[i] = inds[r];
	            inds[r] = temp;
	        }
	    }

	    // now divide the shuffled indices into folds
	    int[][] folds = new int[k][];

	    int foldLength = length / k;
	    int lastFoldLength = length - ((k - 1) * foldLength);

	    for (int ki = 0; ki < k; ki++)
	    {
	        if (ki < k - 1)
	        {
	            folds[ki] = new int[foldLength];
	            System.arraycopy(inds, ki * foldLength, folds[ki], 0, foldLength);
	        }
	        else
	        {
	            folds[ki] = new int[lastFoldLength];
	            System.arraycopy(inds, ki * foldLength, folds[ki], 0, lastFoldLength);
	        }

	        Arrays.sort(folds[ki]);
	    }

	    return folds;
	}
	
	public static HashMap<String,List<Integer>> get_training_test_indices(int[][] folds, int rowDimension, int f){
		List<Integer> test_indices = new ArrayList<Integer>();
		List<Integer> train_indices = new ArrayList<Integer>();
		
		for(int i=0;i<rowDimension;i++){
			for(int j=0;j<folds[f].length;j++){
				if((i == folds[f][j]) && !test_indices.contains(i)){
					test_indices.add(i);
				}
			}
			if(!test_indices.contains(i)){
				train_indices.add(i);
				}
		}

	 HashMap<String,List<Integer>> map =new HashMap();
	 map.put("Test Indices", test_indices);
	 map.put("Train Indices", train_indices);
	 return map;
	}
	
	public static Map<String,double[]> train_data_against_test(HashMap<String,List<Integer>> train_test_indices, Matrix data, Matrix Y){
		//get train data and test data from data
		List<Integer> test_indices = train_test_indices.get("Test Indices");
		List<Integer> train_indices = train_test_indices.get("Train Indices");
				
		double[][] test_data = new double[test_indices.size()][data.getColumnDimension()];
		double[][] train_data = new double[train_indices.size()][data.getColumnDimension()];
		double[][] test_y = new double[test_data.length][1];
		double[][] train_y = new double[train_data.length][1]; 
				
		for(int i=0;i<test_indices.size();i++){
			for(int j=0;j<data.getColumnDimension();j++){
				test_data[i][j] = data.getArray()[test_indices.get(i)][j];
			}
			test_y[i][0] = Y.getArray()[test_indices.get(i)][0];
		}
		
		for(int i=0;i<train_indices.size();i++){
			for(int j=0;j<data.getColumnDimension();j++){
				train_data[i][j] = data.getArray()[train_indices.get(i)][j];
			}
			train_y[i][0] = Y.getArray()[train_indices.get(i)][0];
		}
		
		
		Matrix Train_Data = new Matrix(train_data);
		Matrix Test_Data = new Matrix(test_data);
		Matrix TrainY = new Matrix(train_y);
		Matrix TestY = new Matrix(test_y);
		Matrix XCopy = Train_Data.copy();
		Matrix TestCopy = Test_Data.copy();
	    
	    // transpose
	    Matrix transposeData = XCopy.transpose(); 

	    //Compute X^ 
	    Matrix norm = transposeData.times(Train_Data);
	 
	   //Create an identity matrix of size no. of features 
	    Matrix Icopy = Matrix.identity(Train_Data.getColumnDimension(),Train_Data.getColumnDimension());
	   
	    double[] train_MSE_values = new double[150];
	    double[] test_MSE_values = new double[150];
	    //compute Mean Squared Errors for lambda values from 0 -150
	    for(int lambda=0;lambda<150;lambda++){
	    	Matrix I = Icopy.times(lambda);
	        Matrix Xprod = norm.plus(I);
	        Matrix Xprod_inverse = Xprod.inverse();
	        Matrix XT_y = Xprod_inverse.times(transposeData);
		    Matrix w = XT_y.times(TrainY);
		    double MSE_train_data = L2LinearRegression.compute_MSE(Train_Data,TrainY,w);
		    double MSE_test_data = L2LinearRegression.compute_MSE(Test_Data,TestY,w);
	        train_MSE_values[lambda] = MSE_train_data;
	        test_MSE_values[lambda] =  MSE_test_data;
	    }
	   
	    Map<String,double[]> result = new HashMap<String,double[]>();
	    result.put("MSE for train data", train_MSE_values);
	    result.put("MSE for validation data", test_MSE_values);
	    return result;
	   	  
	}	
}