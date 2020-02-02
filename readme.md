# CS7641 - Assignment One - Supervised Learning

Choosen Datasets:
	https://archive.ics.uci.edu/ml/datasets/Census+Income
	Classify, based on features from the USA 1994 census, whether an individual makes more than
	$50K a year, or not.

	https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018#2016_Financial_Data.csv
	Classify indiviudal stocks by whether to buy, or not buy, based on over 200 features related to
	underlying company performance (e.g. fundemantal data, such as revenue, EPS, EBITDA, etc., as 
	opposed to technical data like SMA, Bollinger Bands, etc.)

## Running Instructions

Each file can be run individually, if desired. Every learner function has the same function signature:

learner(x_data, y_data, data_string)

Where x_data is an ( N x M ) array, with M features, and N data point.
y_data is (N x 1) array, which is the correct classification output, such that each row of y matches x.

*** To generate all images and grid search values, simply run generate_analysis.py (no arguments). ***

### Census Data Images
[Artificial Neural Network](./Code/Census Data ANN Learning Plots.png)
[Decision Tree](./Code/Census Data DTC Learning Plots.png)
[Adaboost with Decision Tree](./Code/Census Data Boosted DTC Learning Plots.png)
[KNN](./Code/Census Data KNN Learning Plots.png)
[SVM](./Code/Census Data SVM Learning Plots.png)

### Stock Data Images
[Artificial Neural Network](./Code/Stock Data ANN Learning Plots.png)
[Decision Tree](./Code/Stock Data DTC Learning Plots.png)
[Adaboost with Decision Tree](./Code/Stock Data Boosted DTC Learning Plots.png)
[KNN](./Code/Stock Data KNN Learning Plots.png)
[SVM](./Code/Stock Data SVM Learning Plots.png)

### Gridsearch results and test predictions
[Results](./Code/results2.txt)
