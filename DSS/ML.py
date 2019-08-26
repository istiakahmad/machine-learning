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


# import warnings filter
# from warnings import simplefilter
# # ignore all future warnings
# simplefilter(action='ignore', category=FutureWarning)

# dataset (csv file) path
path = "crx.csv"


# reading the csv
data = pd.read_csv(path)


def get_features_and_label(data_from_csv):
    data_from_csv.dropna(inplace=True)
    features = data_from_csv.iloc[:, :-1]  # Extract features from the Dataset X1-X23, except the first ID column
    label = data_from_csv.iloc[:, -1]  # Extract the labels from the Dataset.
    return features, label


features, label = get_features_and_label(data)

# print(features)
# print(label)

data.head()
# print(data.describe())
# print((data.groupby('Approved')).size())


# data.hist()
# plt.show()

X = features
y = label


# print(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
                           X, y, test_size = 0.25, random_state = 0)

# print(X.head())
# print('')
# print(y.head())


# Building and Cross-Validating the model
algorithms = []
scores = []
names = []

# algorithms.append(('Logisitic Regression', LogisticRegression()))
# algorithms.append(('K-Nearest Neighbours', KNeighborsClassifier()))
# algorithms.append(('Decision Tree Classifier', DecisionTreeClassifier()))
# algorithms.append(('Random Forest Classifier', RandomForestClassifier()))
# algorithms.append(('Support Vector Classifier', SVC()))
# algorithms.append(('Naive Bias Classifier', GaussianNB()))
# algorithms.append(('Linear Analysis', LinearDiscriminantAnalysis()))
# algorithms.append(('Bagging Classifier', BaggingClassifier()))
# algorithms.append(('GB Classifier', GradientBoostingClassifier()))
# algorithms.append(('ExtraTree Classifier', ExtraTreesClassifier()))
# algorithms.append(('NaiveB Classifier', BernoulliNB()))


for name, algo in algorithms:
    k_fold = model_selection.KFold(n_splits=10, random_state=10)

    # Applying k-cross validation
    cvResults = model_selection.cross_val_score(algo, X_train, y_train,
                                                cv=k_fold, scoring='accuracy')

    scores.append(cvResults)
    names.append(name)
    print(str(name) + ' : ' + str(cvResults.mean()))

# Visually comparing the results of the different algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(scores)
ax.set_xticklabels(names)
plt.show()


# Making predictions and evaluating the predicitons
for name, algo in algorithms:
    clf = algo
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    pred_score = accuracy_score(y_test, y_pred)

    print(str(name) + ' ::::: ' + str(pred_score))
    print('')
    print('Confusion Matrix: ' + str(confusion_matrix(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
