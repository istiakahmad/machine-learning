import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
import time
from sklearn.externals import joblib

start_time = time.time()


def get_header_name(file):
    with open(file, 'rt') as f:
        header = f.readlines(1)[0][:-1].split(',')
    return header


def data_label_encoding(file):
    headers = get_header_name(file)
    data_from_csv = pd.read_csv(file, header=0, index_col=False, names=headers)

    # print(data_from_csv.shape)
    # drop rows with missing values
    data_from_csv.dropna(inplace=True)
    # print('After handling missing value: ')
    # print(data_from_csv.shape)

    for header in headers:
        if data_from_csv[header].dtypes == "object":
            data_from_csv[header] = data_from_csv[header].astype('category')
            data_from_csv[header] = data_from_csv[header].cat.codes
    return data_from_csv


def dataset():
    # dataset (csv file) path

    path = "UCI_Credit_Card.csv"
    # path = "crx.csv"
    # path = "SME.csv"

    # reading the csv
    # data = pd.read_csv(path)
    data = data_label_encoding(path)
    features = data.iloc[:, 1:-1]  # Extract features from the Dataset X1-X23, except the first ID column
    label = data.iloc[:, -1]  # Extract the labels from the Dataset.
    X = features
    y = label
    # print(features)
    # print(label)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


def models():
    # Building and Cross-Validating the model
    algorithms = []
    names = []

    algorithms.append(('GB_Classifier', GradientBoostingClassifier()))
    algorithms.append(('Random_Forest', RandomForestClassifier()))
    algorithms.append(('ExtraTree_Classifier', ExtraTreesClassifier()))
    algorithms.append(('LDA_Classifier', LinearDiscriminantAnalysis()))
    algorithms.append(('KNN_Classification', KNeighborsClassifier()))
    algorithms.append(('ANN_Classification', MLPClassifier()))
    for name, algo in algorithms:
        names.append(name)
    return algorithms, names


def parameters():
    params_RF = {'bootstrap': [True],
                 'max_depth': [10, 12],
                 'max_features': ['sqrt'],
                 'min_samples_leaf': [1, 2],
                 'min_samples_split': [2, 3],
                 'n_estimators': [500, 600, 700]}

    params_GB = {'learning_rate': [0.01, 0.05, 0.1],
                 'n_estimators': [300, 500],
                 'max_depth': [12, 14],
                 'min_samples_split': [2],
                 'min_samples_leaf': [1],
                 'subsample': [1],
                 'max_features': ['sqrt'],
                 'random_state': [10]}

    params_ET = {'bootstrap': [False],
                 'criterion': ['gini', 'entropy'],
                 'max_depth': [18, 19, 20],
                 'max_features': ['sqrt'],
                 'min_samples_leaf': [1],
                 'min_samples_split': [2, 3],
                 'n_estimators': [70, 75, 80]}

    params_LDA = {'solver': ['lsqr'],
                  'shinkage': ['auto'],
                  'n_components': [1, 3, 5]}

    params_ANN = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50)],
                  'activation': ['tanh', 'relu'],
                  'solver': ['sgd', 'adam'],
                  'alpha': [0.0001],
                  'learning_rate': ['constant', 'adaptive'],
                  'random_state': [10],
                  'max_iter': [100, 200]}

    params_KNN = {'n_neighbors': [*range(2, 300, 2)]}

    return params_GB, params_RF, params_ET, params_LDA, params_ANN, params_KNN


def gridsearch():
    params_GB, params_RF, params_ET, params_LDA, params_ANN, params_KNN = parameters()

    # Making predictions and evaluating the predicitons
    algorithms, names = models()
    for name, algo in algorithms:
        clf = algo

        # !!!!!!!!!!!!!!! GB_Classifier !!!!!!!!!!!!!!!!!!!!!
        # if name == 'GB_Classifier':
        #     print(name)
        #     tuning = GridSearchCV(estimator=clf, param_grid=params_GB, scoring='accuracy',
        #                           n_jobs=4,
        #                           iid=False, cv=10)
        #     X_train, X_test, y_train, y_test = dataset()
        #     tuning.fit(X_train, y_train)
        #     best_params = tuning.best_params_
        #     score = tuning.score(X_train, y_train)
        #     y_pred = tuning.predict(X_test)
        #     pred_score = accuracy_score(y_test, y_pred)
        #     # Save the model
        #     model_name = 'GradientBoosting_Classifier_Model.sav'
        #     joblib.dump(tuning, model_name)
        #     # load the model from disk
        #     loaded_model = joblib.load(model_name)
        #     print('Training Score: ', score)
        #     print('Testing Score: ', pred_score)
        #     print(best_params)

        # !!!!!!!!!!!!!!! Random_Forest !!!!!!!!!!!!!!!!!!!!!
        # if name == 'Random_Forest':
        #     print(name)
        #     tuning = GridSearchCV(estimator=clf, param_grid=params_RF, scoring='accuracy',
        #                           n_jobs=4,
        #                           iid=False, cv=10)
        #     X_train, X_test, y_train, y_test = dataset()
        #     tuning.fit(X_train, y_train)
        #     best_params = tuning.best_params_
        #     score = tuning.score(X_train, y_train)
        #     y_pred = tuning.predict(X_test)
        #     pred_score = accuracy_score(y_test, y_pred)
        #
        #     # Save the model
        #     model_name = 'Random_Froest_Model.sav'
        #     joblib.dump(tuning, model_name)
        #
        #     # load the model from disk
        #     loaded_model = joblib.load(model_name)
        #     print('Training Score: ', score)
        #     print('Testing Score: ', pred_score)
        #     print(best_params)

        # !!!!!!!!!!!!!!! ExtraTree_Classifier !!!!!!!!!!!!!!!!!!!!!
        # if name == 'ExtraTree_Classifier':
        #     print(name)
        #     tuning = GridSearchCV(estimator=clf, param_grid=params_ET, scoring='accuracy',
        #                           n_jobs=4,
        #                           iid=False, cv=10)
        #     X_train, X_test, y_train, y_test = dataset()
        #     tuning.fit(X_train, y_train)
        #     best_params = tuning.best_params_
        #     score = tuning.score(X_train, y_train)
        #     y_pred = tuning.predict(X_test)
        #     pred_score = accuracy_score(y_test, y_pred)
        #     # Save the model
        #     model_name = 'ExtraTree_Classifier_Model.sav'
        #     joblib.dump(tuning, model_name)
        #     # load the model from disk
        #     loaded_model = joblib.load(model_name)
        #     print('Training Score: ', score)
        #     print('Testing Score: ', pred_score)
        #     print(best_params)

        # !!!!!!!!!!!!!!! Linear Discrimiat Analysis !!!!!!!!!!!!!!!!!!!!!
        # if name == 'LDA':
        #     print(name)
        #     tuning = GridSearchCV(estimator=clf, param_grid=params_LDA, scoring='accuracy',
        #                           n_jobs=4,
        #                           iid=False, cv=10)
        #     X_train, X_test, y_train, y_test = dataset()
        #     tuning.fit(X_train, y_train)
        #     best_params = tuning.best_params_
        #     score = tuning.score(X_train, y_train)
        #     y_pred = tuning.predict(X_test)
        #     pred_score = accuracy_score(y_test, y_pred)
        #     # Save the model
        #     model_name = 'LinearDiscriminat_Analysis_Model.sav'
        #     joblib.dump(tuning, model_name)
        #     # load the model from disk
        #     loaded_model = joblib.load(model_name)
        #     print('Training Score: ', score)
        #     print('Testing Score: ', pred_score)
        #     print(best_params)

        # !!!!!!!!!!!!!!! Artificial Neural Network !!!!!!!!!!!!!!!!!!!!!
        if name == 'ANN_Classification':
            print(name)
            tuning = GridSearchCV(estimator=clf, param_grid=params_ANN, scoring='accuracy',
                                  n_jobs=4,
                                  iid=False, cv=10)
            X_train, X_test, y_train, y_test = dataset()
            tuning.fit(X_train, y_train)
            best_params = tuning.best_params_
            score = tuning.score(X_train, y_train)
            y_pred = tuning.predict(X_test)
            pred_score = accuracy_score(y_test, y_pred)
            # Save the model
            model_name = name + '_Model.sav'
            joblib.dump(tuning, model_name)
            # load the model from disk
            loaded_model = joblib.load(model_name)
            print('Training Score: ', score)
            print('Testing Score: ', pred_score)
            print(best_params)

        # !!!!!!!!!!!!!!! K-Nearest Neighbors !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if name == 'KNN_Classification':
        #     print(name)
        #     tuning = GridSearchCV(estimator=clf, param_grid=params_KNN, scoring='accuracy',
        #                           n_jobs=4,
        #                           iid=False, cv=10)
        #     X_train, X_test, y_train, y_test = dataset()
        #     tuning.fit(X_train, y_train)
        #     best_params = tuning.best_params_
        #     score = tuning.score(X_train, y_train)
        #     y_pred = tuning.predict(X_test)
        #     pred_score = accuracy_score(y_test, y_pred)
        #     # Save the model
        #     model_name = 'KNN_Model.sav'
        #     joblib.dump(tuning, model_name)
        #     # load the model from disk
        #     loaded_model = joblib.load(model_name)
        #     print('Training Score: ', score)
        #     print('Testing Score: ', pred_score)
        #     print(best_params)

        else:
            pass


gridsearch()
end_time = time.time()
print('Execution Time: ', end_time - start_time)
