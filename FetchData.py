import sqlite3 as sql
import numpy as np
import pickle

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor

def Train_GBM():

    con = sql.connect('DataBase/LandData.db')
    c = con.cursor()  # cursor

    c.execute("SELECT * FROM DataTable;")
    all_result = c.fetchall()

    Train_dataset = np.array(all_result)
    X_train = Train_dataset[:, 1:8]
    Y_train = Train_dataset[:, 0]

    # define the model
    # model = GradientBoostingRegressor(n_estimators = 50, learning_rate = 0.1, subsample = 1, max_depth = 6)
    model = GradientBoostingRegressor()
    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    n_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # report performance
    print('MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    # fit the model on the whole dataset
    model.fit(X_train, Y_train)

    # # save the model to disk
    # filename = 'finalized_model1.sav'
    # pickle.dump(model, open(filename, 'wb'))

Train_GBM()