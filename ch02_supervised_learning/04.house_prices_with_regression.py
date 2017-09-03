# Predykcja cen domów w bostonie
## import
import numpy as np

## pobranie dataset
from sklearn.datasets import load_boston
boston = load_boston()
print (boston.data.shape)
print (boston.feature_names)
print (np.max(boston.target), np.min(boston.target), np.mean(boston.target))
print (boston.DESCR)

##podzielenie zbioru na uczący i testowy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)

## Skalowanie
# Przy regresji skalowanie jest ważne
# Trochę na okrętkę znormalizowane sa wartości y, ale działa ;)
from sklearn.preprocessing import StandardScaler
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train_reshaped) #reshape jest potrzebny

X_train = scalerX.transform(X_train)
y_train = scalery.transform(y_train_reshaped)

X_test = scalerX.transform(X_test)
y_test = scalery.transform(y_test_reshaped)

y_train = y_train.reshape(-1, y_train.shape[0])[0]
y_test = y_test.reshape(-1, y_test.shape[0])[0]



print (np.max(X_train), np.min(X_train), np.mean(X_train), np.max(y_train), np.min(y_train), np.mean(y_train))

##funckja do cross walidacji

from sklearn.model_selection import *
def train_and_evaluate(clf, X_train, y_train):

    clf.fit(X_train, y_train)

    print ("Coefficient of determination on training set:",clf.score(X_train, y_train))

    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))

## model liniowy
from sklearn import linear_model
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None,  random_state=42)
train_and_evaluate(clf_sgd,X_train,y_train)
print (clf_sgd.coef_)

clf_sgd1= linear_model.SGDRegressor(loss='squared_loss', penalty='l2',  random_state=42)
train_and_evaluate(clf_sgd1,X_train,y_train)

clf_sgd2 = linear_model.SGDRegressor(loss='squared_loss', penalty='l1',  random_state=42)
train_and_evaluate(clf_sgd2,X_train,y_train)

clf_sgd3 = linear_model.SGDRegressor(loss='squared_loss', penalty='elasticnet',  random_state=42)
train_and_evaluate(clf_sgd3,X_train,y_train)

clf_ridge = linear_model.Ridge()
train_and_evaluate(clf_ridge,X_train,y_train)

## model vectorowy (vector machines for regresion)
from sklearn import svm
clf_svr= svm.SVR(kernel='linear')
train_and_evaluate(clf_svr,X_train,y_train)

clf_svr_poly= svm.SVR(kernel='poly')
train_and_evaluate(clf_svr_poly,X_train,y_train)

clf_svr_rbf= svm.SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf,X_train,y_train)

clf_svr_poly2= svm.SVR(kernel='poly',degree=2)
train_and_evaluate(clf_svr_poly2,X_train,y_train)


## model Random Forests
from sklearn import ensemble
clf_et=ensemble.ExtraTreesRegressor(n_estimators=10, random_state=42)
train_and_evaluate(clf_et,X_train,y_train)

# print (sorted(zip(clf_et.feature_importances_,boston.feature_names)))

zipped = zip(clf_et.feature_importances_,boston.feature_names)
zipped = list(zipped)
zipped.sort(key=lambda tup: tup[0])

print (list(zipped))

## measure_performance method
from sklearn import metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True, show_r2_score=False):
    y_pred=clf.predict(X)
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")

    if show_classification_report:
        print ("Classification report")
        print (metrics.classification_report(y,y_pred),"\n")

    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y,y_pred),"\n")

    if show_r2_score:
        print ("Coefficient of determination:{0:.3f}".format(metrics.r2_score(y,y_pred)),"\n")

## measure_performance form clasificators
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

clfs = [clf_sgd, clf_sgd1, clf_sgd2, clf_sgd3, clf_ridge, clf_svr, clf_svr_poly, clf_svr_rbf, clf_svr_poly2, clf_et]
for cl in clfs:
    print (namestr(cl, globals()), ' ', cl.__class__)
    measure_performance(X_test,y_test,cl, show_accuracy=False, show_classification_report=False,show_confusion_matrix=False, show_r2_score=True)