import java.io.File;

import java.awt.Color; 
import org.jfree.chart.ChartPanel; 
import org.jfree.chart.JFreeChart; 
import org.jfree.data.xy.XYDataset; 
import org.jfree.data.xy.XYSeries; 
import org.jfree.ui.ApplicationFrame; 
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.XYPlot; 
import org.jfree.chart.ChartFactory; 
import org.jfree.chart.plot.PlotOrientation; 
import org.jfree.data.xy.XYSeriesCollection; 
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.chart.ChartUtilities; 

import java.io.FileInputStream;
import java.io.IOException;

import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import Jama.Matrix;


public class L2LinearRegression {

public static void main(String[] args) throws Exception {
    File train_excel = new File ("C:/Users/BHPriyanka/Desktop/Data Mining/Priyanka_BH_CS6220_HW1/HW1_data/train-100-100.xlsx");
    File test_excel = new File ("C:/Users/BHPriyanka/Desktop/Data Mining/Priyanka_BH_CS6220_HW1/HW1_Data/test-100-100.xlsx");
    FileInputStream fis_train = new FileInputStream(train_excel);
    FileInputStream fis_test = new FileInputStream(test_excel);

    XSSFWorkbook wb_train = new XSSFWorkbook(fis_train);
    XSSFWorkbook wb_test = new XSSFWorkbook(fis_test);
    XSSFSheet train_sheet = wb_train.getSheetAt(0);
    XSSFSheet test_sheet = wb_test.getSheetAt(0);
    Map<String,double[]> result = compute_w(train_sheet, test_sheet);
    double[] train_MSE_values = result.get("MSE for train data");
    double[] test_MSE_values = result.get("MSE for validation data");
    for(int i=0;i<150;i++){
    	System.out.println(train_MSE_values[i] +" " + test_MSE_values[i]);
    }
    XYPlot_MSE_lambda(150, train_MSE_values, test_MSE_values);
 }

	public static Map<String,double[]> compute_w(XSSFSheet train_sheet, XSSFSheet test_sheet){
		Matrix train_X = new Matrix(getMatrix(train_sheet));
		Matrix test_X = new Matrix(getMatrix(test_sheet));

	    Matrix TRAINX = modify_matrix(train_X); //remove last column and add first column of 1s
	    Matrix TESTX = modify_matrix(test_X);

	    //get the value of y for all data points
	    Matrix TRAINY = train_X.getMatrix(1, train_X.getRowDimension() - 1, train_X.getColumnDimension() - 1, train_X.getColumnDimension() - 1);
	    Matrix TESTY = test_X.getMatrix(1, test_X.getRowDimension() - 1, test_X.getColumnDimension() - 1, test_X.getColumnDimension() - 1);
			      
	    Matrix TrainXCopy = TRAINX.copy();
	    Matrix TestXCopy = TESTX.copy();
	    
	    // transpose
	    Matrix transposeData = TrainXCopy.transpose(); 

	    //Compute X^ 
	    Matrix norm = transposeData.times(TRAINX);
	 
	   //Create an identity matrix of size no. of features 
	    Matrix Icopy = Matrix.identity(TRAINX.getColumnDimension(),TRAINX.getColumnDimension());
	    double[] TrainMSE_values= new double[151];
	    double[] TestMSE_values= new double[151];
	    
	    //compute Mean Squared Errors for lambda values from 0 -150
	    for(int lambda=1;lambda<=150;lambda++){
	    	Matrix I = Icopy.times(lambda);
	    	Matrix Xprod = norm.plus(I);
	        Matrix Xprod_inverse = Xprod.inverse();
	        Matrix XT_y = Xprod_inverse.times(transposeData);
	        Matrix w = XT_y.times(TRAINY);
	        double TrainMSE = compute_MSE(TRAINX,TRAINY,w);
	        double TestMSE = compute_MSE(TESTX,TESTY,w);
	        TrainMSE_values[lambda] = TrainMSE;
	        TestMSE_values[lambda] = TestMSE;
	    }
	    
	    Map<String,double[]> result = new HashMap<String,double[]>();
	    result.put("MSE for train data", TrainMSE_values);
	    result.put("MSE for validation data", TestMSE_values);
	    return result;
	}

	
	public static double[][] getMatrix(XSSFSheet sheet){
		int rowNum = sheet.getLastRowNum()+1;
	    int colNum = sheet.getRow(0).getLastCellNum();
	    Iterator rows = sheet.rowIterator();
	    double x[][] = new double[rowNum][colNum];
	    while(rows.hasNext()){
	    	XSSFRow row = (XSSFRow)rows.next();
	    	Iterator cells = row.cellIterator ();
	    	while (cells.hasNext())
	    	{
	    		XSSFCell cell = (XSSFCell)cells.next();
	    		if(row.getRowNum() > 0){
	    			Double value = cell.getNumericCellValue();
	       			x[row.getRowNum()][cell.getColumnIndex()] = value;
	    		}
	       }
	    }
	    return x;
	}
	
	
	public static Matrix modify_matrix(Matrix X){    
	    Matrix data = X.getMatrix(1, X.getRowDimension() - 1, 0, X.getColumnDimension() - 2);
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
	
	public static double compute_MSE(Matrix X, Matrix Y, Matrix w){
	    //	double MSE= (X.times(w).minus(Y)).norm2()/ rowNum;
	    //	return MSE;
	    	double error = 0.0;
			int row = X.getRowDimension();
			int column = X.getColumnDimension();
			assert row == Y.getRowDimension();
			assert column == w.getColumnDimension();
			
			Matrix predictTargets = predict(X, w);
			for (int i = 0; i < row; i++) {
				error = error + (Y.get(i, 0) - predictTargets.get(i, 0)) * (Y.get(i, 0) - predictTargets.get(i, 0));
			}

			double res = (1/(double)row)* error;
			return res;
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

	 

 
   	public static void XYPlot_MSE_lambda(int lambda, double[] train_MSE_values, double[] test_MSE_values){
    		JFreeChart xylineChart = ChartFactory.createXYLineChart(
    		         "XY Plot of MSE vs lambda" ,
    		         "XAxis" ,
    		         "YAxis" ,
    		         createDataset(lambda, train_MSE_values,test_MSE_values) ,
    		         PlotOrientation.VERTICAL ,
    		         false , false , false);
    		
    		ChartPanel chartPanel = new ChartPanel( xylineChart );
    	    chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
    	    final XYPlot plot = xylineChart.getXYPlot( );
    	    XYDotRenderer renderer = new XYDotRenderer( );
    	       	    
    	    renderer.setSeriesPaint( 0 , Color.RED );  //train data is red
    	    renderer.setSeriesPaint( 1 , Color.BLUE );  //test data is blue
    	    
    	    renderer.setDotWidth(4);
    	    renderer.setDotHeight(5);
    	    
    	    /*NumberAxis domain = (NumberAxis) plot.getDomainAxis();
    	    domain.setRange(1, 150);
    	    domain.setTickUnit(new NumberTickUnit(10));
    	    domain.setVerticalTickLabels(true);
    	    NumberAxis range = (NumberAxis) plot.getRangeAxis();
    	    range.setRange(0.0, 2000);
    	    range.setTickUnit(new NumberTickUnit(20*/
    	    
    	    plot.setRenderer( renderer ); 
    	    plot.setBackgroundPaint(Color.LIGHT_GRAY);
    	    plot.setRangeGridlinesVisible(true);
    	    plot.setRangeGridlinePaint(Color.BLACK);
    	    ApplicationFrame a =  new ApplicationFrame("XYPlot");
    	    a.setContentPane( chartPanel );
    	    
    	    
    	    File imageFile = new File("./XYChart.png");
    	    int width = 1200;
    	    int height = 800;
    	     
    	    try {
    	        ChartUtilities.saveChartAsPNG(imageFile, xylineChart, width, height);
    	    } catch (IOException ex) {
    	        System.err.println(ex);
    	    }
    	}
    	
    	public static XYDataset createDataset(int lambda, double[] train_MSE_values, double[] test_MSE_values )
    	   {
    	      final XYSeries series1 = new XYSeries( "Train Data" );
    	      final XYSeries series2 = new XYSeries( "Test Data" );
    	      
    	      for(int i=0;i<lambda;i++){
    	    	  series1.add(i, train_MSE_values[i]);
    	    	  series2.add(i, test_MSE_values[i]);
    	      }
    	         
    	      final XYSeriesCollection dataset = new XYSeriesCollection( );          
    	      dataset.addSeries( series1 );
    	      dataset.addSeries(series2);
    	      return dataset;
    	   }
    	 
      }


