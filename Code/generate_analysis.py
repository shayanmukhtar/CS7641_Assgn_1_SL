import ann_learning
import boosted_learning
import decision_tree_learning
import process_data
import knn_learning
import support_vector_learning
import time


def main():
    timer = time.time()

    x_gas_data, y_gas_data = process_data.process_gas_data('./../Datasets/Gas_Data')
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016", "2017"])

    print("\n---------- Running Data on Decision Tree Learner -------------\n")
    decision_tree_learning.run_decision_tree(x_census_data, y_census_data, "Census Data")
    decision_tree_learning.run_decision_tree(x_stock_data, y_stock_data, "Stock Data")

    print("\n---------- Running Data on Artificial Neural Network Learner -------------\n")
    ann_learning.neural_network_learning(x_census_data, y_census_data, "Census Data")
    ann_learning.neural_network_learning(x_stock_data, y_stock_data, "Stock Data")

    print("\n---------- Running Data on Decision Tree Learner with Adaboost -------------\n")
    boosted_learning.run_boosted_learning(x_census_data, y_census_data, "Census Data")
    boosted_learning.run_boosted_learning(x_stock_data, y_stock_data, "Stock Data")

    print("\n---------- Running Data on Support Vector Learner -------------\n")
    support_vector_learning.run_svc_learning(x_census_data, y_census_data, "Census Data")
    support_vector_learning.run_svc_learning(x_stock_data, y_stock_data, "Stock Data")

    print("\n---------- Running Data on KNN Learner -------------\n")
    knn_learning.run_knn_learning(x_census_data, y_census_data, "Census Data")
    knn_learning.run_knn_learning(x_stock_data, y_stock_data, "Stock Data")

    print("Total Execution time: " + str(time.time() - timer) + "s")


if __name__ == '__main__':
    main()