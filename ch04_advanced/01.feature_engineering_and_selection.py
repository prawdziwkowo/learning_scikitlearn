## load titanic data with pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titanic = pd.read_csv('titanic.csv')
print (titanic)
# titanic = titanic.drop(['embarked'], axis=1) #po wywaleniu embarced algorytm lepiej działa
print (titanic.head()[['pclass', 'survived', 'age', 'boat', 'sex']])
titanic.describe()

## dodanie nowych kolumn (binarnych)
from sklearn import feature_extraction

def one_hot_dataframe(data, cols, replace=False):
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)

titanic, titanic_n = one_hot_dataframe(titanic, ['pclass', 'sex', 'embarked'], replace=True)

titanic, titanic_n = one_hot_dataframe(titanic, ['home.dest', 'room', 'ticket', 'boat'], replace=True)

titanic.describe()

## uzupełnienie danych gdzie ich nie ma (wiek na średnią - reszta na 0)
mean = titanic['age'].mean()
titanic['age'].fillna(mean, inplace=True)
titanic.fillna(0, inplace=True)
titanic['age'].describe()

## dane treningowe i testowe
from sklearn.model_selection import train_test_split
titanic_target = titanic['survived']
titanic_data = titanic.drop(['name', 'row.names', 'survived'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(titanic_data, titanic_target, test_size=0.25, random_state=33)

## decission tree
from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(X_train, y_train)

from sklearn import metrics
y_pred = dt.predict(X_test)
print ("Accuracy:{0:.3f}" . format(metrics.accuracy_score(y_test, y_pred)), "\n")

## features_selection Percentile (percentile - ile % kolumn brać pod uwagę)
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
print (titanic_data.columns[fs.get_support()])
print (fs.scores_[2])
print (titanic_data.columns[2])

## uczenie za pomoca powyższego
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
y_pred_fs = dt.predict(X_test_fs)
print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test, y_pred_fs)),"\n")

## znalezienie ilości najlepszych percentili za pomoca brute force
from sklearn.model_selection import cross_val_score

percentiles = range(1, 100, 5)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    print (i,scores.mean())
    results = np.append(results, scores.mean())

optimal_percentil = np.where(results == results.max())[0][0]

print ("Optimal number of features:{0}".format(percentiles[optimal_percentil]), "\n")

## wykres powyższego
import pylab as pl
pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation accuracy)")
pl.plot(percentiles,results)
print ("Mean scores:",results)

##wiemy ile jest najlepiej optymalnych percentili - więc badamy - no i wychodzi nam najlepiej
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=percentiles[optimal_percentil])
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
y_pred_fs = dt.predict(X_test_fs)
print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test, y_pred_fs)),"\n")

## GINI
dt = tree.DecisionTreeClassifier(criterion='entropy')
scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
print ("Entropy criterion accuracy on cv: {0:.3f}".format(scores.mean()))
dt = tree.DecisionTreeClassifier(criterion='gini')
scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
print ("Gini criterion accuracy on cv: {0:.3f}".format(scores.mean()))

dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
y_pred_fs = dt.predict(X_test_fs)
print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y_test, y_pred_fs)),"\n")