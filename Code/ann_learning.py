import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import process_data
import matplotlib.pyplot as plt
import evaluate_model_learning_complexity

# class NeuralNetLearner(object):
#     def __init__(self, input_dim, output_dim, alpha=0.01, hidden_layer_size=16):
#         # build a stack of layers into a feedforward neural net
#         self.neural_net = Sequential()
#         self.neural_net.add(Dense(hidden_layer_size, activation='relu', input_dim=input_dim))
#         self.neural_net.add(Dense(hidden_layer_size, activation='relu'))
#         self.neural_net.add(Dense(output_dim, activation='linear'))
#         self.neural_net.compile(loss='mse', optimizer=Adam(lr=alpha))


def neural_network_learning(x_data, y_data, data_string=""):
    # split into training set and validation set
    train_size = 0.6
    random_state = 86
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=random_state)

    # create the grid search parameter dictionary
    paramaters = [
        {'solver': ['adam'], 'hidden_layer_sizes': [(16, 16), (32, 32), (64, 64)]},
        {'solver': ['sgd'],  'hidden_layer_sizes': [(16, 16), (32, 32), (64, 64)]},
    ]

    grid_searcher = GridSearchCV(MLPClassifier(), paramaters)

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
    title = data_string + " ANN - " + str(grid_searcher.best_params_)
    alpha_range = np.logspace(-6, -1, 5)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train, y_train, parameter="alpha",
                                                                    param_grid=alpha_range, param_string="Alpha",
                                                                    log_range=True)

    figure.savefig(data_string + " ANN Learning Plots")
    score_model_test_data(x_test, y_test, grid_searcher.best_estimator_, str(grid_searcher.best_params_), data_string)


def score_model_test_data(x_test, y_test, estimator, param_string, data_string):
    print("Scoring ANN with parameters: " + param_string + "\tOn Data: " + data_string)
    print(estimator.score(x_test, y_test))
    print()


def score_gas_data(x_train, y_train, x_test, y_test):
    print("ANN Model Score on " + "Gas Data" + " test data:")

    dtc = MLPClassifier(hidden_layer_sizes=(32, 32), solver="adam", alpha=1e-5)
    dtc.fit(x_train, y_train)
    print(str(dtc.score(x_test, y_test)))
    print()


def score_stock_data(x_train, y_train, x_test, y_test):
    print("ANN Model Score on " + "Stock Data" + " test data:")

    dtc = MLPClassifier(hidden_layer_sizes=(32, 32), solver='sgd', alpha=1e-6)
    dtc.fit(x_train, y_train)
    print(str(dtc.score(x_test, y_test)))
    print()


def main():
    x_gas_data, y_gas_data = process_data.process_gas_data('./../Datasets/Gas_Data')
    x_bank_data, y_bank_data = process_data.process_bank_data('./../Datasets/Banking')
    x_letter_data, y_letter_data = process_data.process_letter_data('./../Datasets/Letter')
    x_spam_data, y_spam_data = process_data.process_spam_data('./../Datasets/Spam')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016"])
    neural_network_learning(x_gas_data, y_gas_data, "Gas Sensor Data")
    # neural_network_learning(x_bank_data, y_bank_data, "Bank Data")
    # neural_network_learning(x_letter_data, y_letter_data, "Letter Data")
    # neural_network_learning(x_spam_data, y_spam_data, "Spam Data")
    neural_network_learning(x_stock_data, y_stock_data, "Stock Data")


if __name__ == '__main__':
    main()