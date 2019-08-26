import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
import time

start_time = time.time()


def dataset():
    # dataset (csv file) path
    path = "UCI_Credit_Card.csv"
    # reading the csv
    data = pd.read_csv(path)
    features = data.iloc[:, 1:-1]  # Extract features from the Dataset X1-X23, except the first ID column
    label = data.iloc[:, -1]  # Extract the labels from the Dataset.
    X = features
    y = label
    # print(features)
    # print(label)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


params = {'learning_rate':[0.01],
           'n_estimators':[500],
           'max_depth': [12],
           'min_samples_split':[2],
           'min_samples_leaf':[1],
           'subsample':[1],
           'max_features':['sqrt'],
           'random_state':[10]}


def gridsearch(params):
    tuning = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=params, scoring='accuracy', n_jobs=4, iid=False, cv=5)
    X_train, X_test, y_train, y_test = dataset()
    tuning.fit(X_train, y_train)
    best_params = tuning.best_params_
    score = tuning.score(X_train, y_train)
    print(score)
    print(best_params)
    print(tuning.best_params_)


gridsearch(params)
end_time = time.time()
print('Execution Time: ', end_time - start_time)


