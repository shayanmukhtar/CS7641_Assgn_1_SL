from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import process_data
import evaluate_model_learning_complexity

def run_svc_learning(x_data, y_data, data_string=""):
    # split into training set and validation set
    train_size = 0.6
    random_state = 86
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=random_state)

    # create the grid search parameter dictionary

    paramaters = [
        {'kernel': ['rbf'],     'C': [1, 10, 100, 1000]},
        {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['poly'],    'C': [1, 10, 100, 1000]}
    ]

    grid_searcher = GridSearchCV(svm.SVC(), paramaters)

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
    title = data_string + " SVM - " + str(grid_searcher.best_params_)
    gamma_range = np.logspace(-6, -1, 5)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train, y_train, parameter="gamma",
                                                                    param_grid=gamma_range, param_string="Gamma",
                                                                    log_range=True)

    figure.savefig(data_string + " SVM Learning Plots")
    score_model_test_data(x_test, y_test, grid_searcher.best_estimator_, str(grid_searcher.best_params_), data_string)


def score_model_test_data(x_test, y_test, estimator, param_string, data_string):
    print("Scoring SVM with parameters: " + param_string + "\tOn Data: " + data_string)
    print(estimator.score(x_test, y_test))
    print()

def score_gas_data(x_train, y_train, x_test, y_test):
    print("SVM Model Score on " + "Gas Data" + " test data:")

    svc = svm.SVC(C=1000, kernel="rbf", gamma=1e-4)
    svc.fit(x_train, y_train)
    print(str(svc.score(x_test, y_test)))
    print()


def score_stock_data(x_train, y_train, x_test, y_test):
    print("SVM Model Score on " + "Stock Data" + " test data:")

    svc = svm.SVC(C=10, kernel="poly", gamma=0.1)
    svc.fit(x_train, y_train)
    print(str(svc.score(x_test, y_test)))
    print()


def main():
    x_gas_data, y_gas_data = process_data.process_gas_data('./../Datasets/Gas_Data')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016"])
    run_svc_learning(x_stock_data, y_stock_data, "Stock Data")
    run_svc_learning(x_gas_data, y_gas_data, "Gas Sensor Data")


if __name__ == '__main__':
    main()