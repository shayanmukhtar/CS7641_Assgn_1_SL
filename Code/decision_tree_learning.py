import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import process_data
from sklearn.model_selection import GridSearchCV
import process_data
import evaluate_model_learning_complexity


def run_decision_tree(x_data, y_data, data_string=""):
    # split into training set and validation set
    train_size = 0.6
    random_state = 86
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=random_state)

    print("Output Classification Description of " + data_string)
    unique, counts = np.unique(y_data, return_counts=True)
    for element in range(0, len(unique)):
        print("Element: " + str(unique[element]) + "\t\t" + str(100 * counts[element] / len(y_data)) + "%")

    print()

    # create the grid search parameter dictionary
    paramaters = [
        {'splitter': ['best'],   'min_samples_split': [2, 4, 8, 16]},
        {'splitter': ['random'], 'min_samples_split': [2, 4, 8, 16]},
    ]

    grid_searcher = GridSearchCV(DecisionTreeClassifier(criterion="entropy"), paramaters)

    grid_searcher.fit(x_train, y_train)

    # form a 2d list of your data
    report = [["Parameters", "Mean Fit Time", "Std Dev Fit Time", "Split 0 Score", "Split 1 Score", "Split 2 Score", "Split 3 Score", "Split 4 Score"]]
    for row in range(0, len(grid_searcher.cv_results_['params'])):
        row_data = [str(grid_searcher.cv_results_['params'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['mean_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['std_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split0_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split1_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split2_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split3_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split4_test_score'][row]),
                    ]
        report.append(row_data)

    # print dictionary of scores
    print("Grid Search Report")
    print()

    col_width = max(len(word) for row in report for word in row) + 2  # padding
    for row in report:
        print("".join(word.ljust(col_width) for word in row))

    # plot the learning curve
    title = data_string + " DTC - " + str(grid_searcher.best_params_)
    max_depth_range = range(1, 25, 1)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train, y_train, parameter="max_depth",
                                                                    param_grid=max_depth_range, param_string="Max Tree Depth",
                                                                    log_range=False)

    figure.savefig(data_string + " DTC Learning Plots")
    score_model_test_data(x_test, y_test, grid_searcher.best_estimator_, str(grid_searcher.best_params_), data_string)


def score_model_test_data(x_test, y_test, estimator, param_string, data_string):
    print("Scoring DTC with parameters: " + param_string + "\tOn Data: " + data_string)
    print(estimator.score(x_test, y_test))
    print()


def score_census_data(x_train, y_train, x_test, y_test):
    print("DTC Model Score on " + "Gas Data" + " test data:")

    dtc = DecisionTreeClassifier(min_samples_split=16, splitter='random', max_depth=7)
    dtc.fit(x_train, y_train)
    print(str(dtc.score(x_test, y_test)))
    print()


def score_gas_data(x_train, y_train, x_test, y_test):
    print("DTC Model Score on " + "Gas Data" + " test data:")

    dtc = DecisionTreeClassifier(min_samples_split=2, splitter='best', max_depth=12)
    dtc.fit(x_train, y_train)
    print(str(dtc.score(x_test, y_test)))
    print()


def score_stock_data(x_train, y_train, x_test, y_test):
    print("DTC Model Score on " + "Stock Data" + " test data:")

    dtc = DecisionTreeClassifier(min_samples_split=8, splitter='random', max_depth=2)
    dtc.fit(x_train, y_train)
    print(str(dtc.score(x_test, y_test)))
    print()


def main():
    x_gas_data, y_gas_data = process_data.process_gas_data('./../Datasets/Gas_Data')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016"])
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    # run_decision_tree(x_gas_data, y_gas_data, "Gas Sensor Data")
    # run_decision_tree(x_stock_data, y_stock_data, "Stock Data")
    run_decision_tree(x_census_data, y_census_data, "Census Data")


if __name__ == '__main__':
    main()
