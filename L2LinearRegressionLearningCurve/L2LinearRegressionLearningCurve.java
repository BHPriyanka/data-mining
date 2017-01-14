import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.io.FileInputStream;
import java.io.IOException;

import java.awt.Color; 
import org.jfree.chart.ChartPanel; 
import org.jfree.chart.JFreeChart; 
import org.jfree.data.xy.XYDataset; 
import org.jfree.data.xy.XYSeries; 
import org.jfree.ui.ApplicationFrame; 
import org.jfree.chart.plot.XYPlot; 
import org.jfree.chart.ChartFactory; 
import org.jfree.chart.plot.PlotOrientation; 
import org.jfree.data.xy.XYSeriesCollection; 
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ChartUtilities; 

import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import java.util.Iterator;
import Jama.Matrix;


public class L2LinearRegressionLearningCurve {

	public static int lambda = 1; //1,46,150
	
	public static void main(String[] args) throws Exception {
    File train_excel = new File ("C:/Users/BHPriyanka/Desktop/Data Mining/Priyanka_BH_CS6220_HW1/HW1_Data/train-1000-100.xlsx");
    FileInputStream fis_train = new FileInputStream(train_excel);
    XSSFWorkbook wb_train = new XSSFWorkbook(fis_train);
    XSSFSheet train_sheet = wb_train.getSheetAt(0);
    
    //compute the MSE for different subset sizes ranging from 50 till 1000
    ArrayList<double[]> result = new ArrayList<double[]>();   
    for(int set=50;set<1000;set+=50){
    	result.add(compute_ten_times(train_sheet, set));
    }
    
    XYPlot_MSE_size(result);
    }

	public static double[] compute_w(XSSFSheet train_sheet,int size){
		double[][] x = L2LinearRegression.getMatrix(train_sheet);
	   
		//Generate random subsets of size(returns the train data- subset of size and test data
		//test data = total data points-train data
	    Matrix[] res = generate_random_subsets(size, x);
	    Matrix Xnew = res[0];
	 	Matrix TESTX = res[1];
	 	 	
	 	Matrix TRAINX = L2LinearRegression.modify_matrix(Xnew);
	 	    
	    Matrix Y = Xnew.getMatrix(1, Xnew.getRowDimension() -1, Xnew.getColumnDimension() -1, Xnew.getColumnDimension() -1);
	       
	    // transpose
	    Matrix XCopy = TRAINX.copy();
	    Matrix transposeData = XCopy.transpose(); 

	    //Compute X^ 
	    Matrix norm = transposeData.times(TRAINX);
  	    
	   //Create an identity matrix of size no. of features 
	    Matrix Icopy = Matrix.identity(TRAINX.getColumnDimension(),TRAINX.getColumnDimension());
	    Matrix I= Icopy.times(lambda);
	   
	    //compute w
	    Matrix Xprod  = norm.plus(I);
	    Matrix Xprod_inverse = Xprod.inverse();
	    Matrix XT_y = Xprod_inverse.times(transposeData);
	    Matrix w = XT_y.times(Y);
	   	    	    
	    double MSE_train_data = L2LinearRegression.compute_MSE(TRAINX,Y,w);
	    
	    //Then compute MSEs for the test data obtained using w calculated above
	    Matrix TEST_X = modify_matrixTest(TESTX);
	  	Matrix TEST_Y = TESTX.getMatrix(0, TESTX.getRowDimension() - 1, TESTX.getColumnDimension() - 1, TESTX.getColumnDimension() - 1);
	  	    
	    double MSE_test_data = L2LinearRegression.compute_MSE(TEST_X,TEST_Y,w);
	    
	    //return the result as an array of both  MSEs of train data and test data
	    double[] result = new double[2];
	    result[0] = MSE_train_data;
	    result[1]= MSE_test_data;
	    return result;
	}

    
	private static Matrix modify_matrixTest(Matrix X){
	    Matrix data = X.getMatrix(0, X.getRowDimension() - 1, 0, X.getColumnDimension() - 2);
		int row = data.getRowDimension();
		int col = data.getColumnDimension() + 1;
		Matrix modifiedData = new Matrix(row, col);
		for (int r = 0; r < row; ++r) {
			for (int c = 0; c < col; ++c) {
				if (c == 0) {
					modifiedData.set(r, c, 1.0);
				} else {
					modifiedData.set(r, c, data.get(r, c-1));
				}
			}
		}
		return modifiedData;
	}
	
	    private static Matrix predict(Matrix data, Matrix weights) {
			int row = data.getRowDimension();
			Matrix predictTargets = new Matrix(row, 1);
			for (int i = 0; i < row; i++) {
				double value = multiply(data.getMatrix(i, i, 0, data.getColumnDimension() -1 ), weights);
				//System.out.println("value:" +value);
				predictTargets.set(i, 0, value);
			}
			return predictTargets;
		}
	    
		private static Double multiply(Matrix data, Matrix weights) {
			Double sum = 0.0;
			int column = data.getColumnDimension();
			for (int i = 0; i <column; i++) {
				sum += data.get(0, i) * weights.get(i, 0);
			}
			return sum;
		}

 
   	public static void XYPlot_MSE_size(ArrayList<double[]> train_MSE_values){
    		JFreeChart xylineChart = ChartFactory.createXYLineChart(
    		         "XY Plot of MSE vs number of data points" ,
    		         "XAxis" , // number of data points
    		         "YAxis" , //MSE
    		         createDataset(train_MSE_values) ,
    		         PlotOrientation.VERTICAL ,
    		         false , false , false);
    		
    		ChartPanel chartPanel = new ChartPanel( xylineChart );
    	    chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
    	    final XYPlot plot = xylineChart.getXYPlot( );
    	    XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer( );
    	       	    
    	    renderer.setSeriesPaint( 0 , Color.RED ); //traindata
    	    renderer.setSeriesPaint( 1 , Color.BLUE ); //test data
    	     	    
    	    plot.setRenderer( renderer ); 
    	    plot.setBackgroundPaint(Color.LIGHT_GRAY);
    	    plot.setRangeGridlinesVisible(true);
    	    plot.setRangeGridlinePaint(Color.BLACK);
    	    ApplicationFrame a =  new ApplicationFrame("XYPlot");
    	    a.setContentPane( chartPanel );
    	    
    	    
    	    File imageFile = new File("./XYLearningCurve_150.png");
    	    int width = 1000;
    	    int height = 800;
    	     
    	    try {
    	        ChartUtilities.saveChartAsPNG(imageFile, xylineChart, width, height);
    	    } catch (IOException ex) {
    	        System.err.println(ex);
    	    }
    	}
    	
    	public static XYDataset createDataset(ArrayList<double[]> MSE_values)
    	   {
    	      final XYSeries series1 = new XYSeries( "Linear Curve for train data" );
    	      final XYSeries series2 = new XYSeries( "Linear Curve for test data" );
 
    	      int count=50;
    	      for(double[] data : MSE_values){
    	    		 series1.add(count, data[0]);
    	    		 series2.add(count, data[1]);
    	    		 count+=50;
    	      }    	         
    	      
    	      final XYSeriesCollection dataset = new XYSeriesCollection( );          
    	      dataset.addSeries( series1 );
    	      dataset.addSeries( series2 );
    	      return dataset;
    	   }
    	
    	
    	public static void swapRows(double[][] a, int row0, int row1) {
    		    int cols = a[0].length;
    		    for (int col=0; col<cols; col++)
    		      swap(a, row0, col, row1, col);
    		  }
    	public static void swap(double[][] a, int i0, int j0, int i1, int j1) {
    		    double temp = a[i0][j0];
    		    a[i0][j0] = a[i1][j1];
    		    a[i1][j1] = temp;
    		  }
     	
    	public static Matrix[] generate_random_subsets(int size, double[][] data){
    		Random rand = new Random();
    	    	   		
    		for(int i=0;i<size;i++){
    	    	int  m = rand.nextInt(size) + 1;
    	    	int  n = rand.nextInt(size) + 1;
    	    	swapRows(data, m, n);
    		}
    		
    		Matrix Data = new Matrix(data);
      	    Matrix new_data = Data.getMatrix(size+1, Data.getRowDimension()-1, 0, Data.getColumnDimension()-1);
      	       	          	    
      	    Matrix Subset = Data.getMatrix(1, size, 0, Data.getColumnDimension()-1);
    		
      	    Matrix[] array_matrices = new Matrix[2];
    		array_matrices[0] = Subset;
    		array_matrices[1]= new_data;
    		return array_matrices;
    	}
    	
    	public static double compute_average(double[] MSE_values){
    		double sum =0.0;
    		for(int i=0;i<MSE_values.length;i++){
    			sum = sum + MSE_values[i];
    		}
    		return sum/MSE_values.length;
    	}
    	
    	public static double[] compute_ten_times(XSSFSheet train_sheet, int set){
    	    double[] train_MSE_values = new double[10];
    	    double[] test_MSE_values = new double[10];
    		double[] result = new double[2];
    		for(int j=0;j<10;j++){
            	result =  compute_w(train_sheet, set);
               	train_MSE_values[j] = result[0];
               	test_MSE_values[j] = result[1];
            }
    	     double avg_train_MSE_values = compute_average(train_MSE_values);
    	     double avg_test_MSE_values = compute_average(test_MSE_values);
    	
    	double[] MSE_train_test = new double[2];
    	MSE_train_test[0] = avg_train_MSE_values;
    	MSE_train_test[1] = avg_test_MSE_values;
    	return MSE_train_test;
      }
}

