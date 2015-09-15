import sys
import numpy as np
from sklearn import linear_model
from preference_data_reader import FlixsterDataReader

def rmse(arr1, arr2):
    return np.sqrt(np.mean((pred_target - flixster.test_target)**2))

def fit_model(model_name):
    reg = None
    if model_name == "linear_regression":
        reg = linear_model.LinearRegression()
        reg.fit(flixster.train_data, flixster.train_target)
    elif model_name == "ridge_regression":
        reg = linear_model.Ridge(alpha=0.5)
        reg.fit(flixster.train_data, flixster.train_target)
    elif model_name in ("user_average", "item_average"):
        reg = linear_model.Ridge(alpha=0.5)
        reg.fit(flixster.train_data, flixster.train_target)

    return reg

if __name__ == "__main__":
    model_name = sys.argv[1]
    input_filepath = "flixster/ratings_small.txt"
    print "Reading data... ",
    flixster = FlixsterDataReader(input_filepath, model_name)
    print "Done"

    print "Creating train test split... ",
    flixster.create_train_test_split(split_fraction=0.8)
    print "Done"

    print "Fitting model... ",
    reg = fit_model(model_name)
    print "Done"

    pred_target = reg.predict(flixster.test_data)
    print "Root mean squared error", rmse(pred_target, flixster.test_target)
