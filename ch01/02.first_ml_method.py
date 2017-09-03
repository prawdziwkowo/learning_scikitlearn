# Próba predykcji co to za kwiatek na podstawie rozmiaru działki kielicha (takiej zielonej osłonki na dole kwiatu)
# klasyfikacja liniowa
# W datasetsach mamy też zapisana wielkość płatka
# okazuje sie z wykresu, że nie ma podziału pomiędzy versicolor i verginca dlatego klasyfikacja binarna
# czy jest to setosa czy nie

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X_iris, Y_iris = iris.data, iris.target

# Get dataset with only the first two attributes
X, y = X_iris[:,:2], Y_iris
# Split the dataset into a trainig and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print (X_train.shape, y_train.shape) # wysietlenie rozmiaru obiektów
# Standarize the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#rysowanie wykresu na podstawie danych
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    px = X_train[:, 0][y_train == i]
    py = X_train[:, 1][y_train == i]
    plt.scatter(px, py, c=colors[i])

plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

#pokazanie normalnie o tej porze wykresu
# plt.show()

# create the linear model classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
# fit (train) the classifier
clf.fit(X_train, y_train)
# print learned coeficients
print (clf.coef_)
print (clf.intercept_)

#Rysuje 3 wykreasy z liniami podzialu (bo mamy aktyalnie 3 rodzaje roślinek)
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
xs = np.arange(x_min,x_max,0.5)
fig, axes = plt.subplots(1,3)
fig.set_size_inches(10,6)
for i in [0,1,2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class ' + str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i]) #dodaje wykres do obrazka
    for j in range(len(colors)):
        px = X_train[:, 0][y_train == j]
        py = X_train[:, 1][y_train == j]
        plt.scatter(px, py, c=colors[j])
    ys = (-clf.intercept_[i]-xs*clf.coef_[i,0])/clf.coef_[i,1]
    plt.plot(xs,ys)
# plt.show()

#predykcja już konkretnego kwiatka
print (clf.predict(scaler.transform([[4.7, 3.1]])))
print (clf.decision_function(scaler.transform([[4.7, 3.1]])))

#sprawdzamy jak mocno nasza predykcja działa (wychodzi, że 70-80%) dla danych
#ale to jest z czapy, bo sprawdzamy na danych treningowych
from sklearn import metrics
y_train_pred = clf.predict(X_train)
print (metrics.accuracy_score(y_train, y_train_pred))

#tutaj sprawdzimy na danych testowych
y_pred = clf.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))

#statystyki
print (metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
#matryca klasyfikacji - fajne bo można zobaczyć blędy klasyfikacji
print (metrics.confusion_matrix(y_test, y_pred))



from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

# create a composite estimator made by a pipeline of the standarization and the linear model
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_model', SGDClassifier())
])
# create a k-fold croos validation iterator of k=5 folds
cv = KFold(5, shuffle=True, random_state=33)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf, X, y, cv=cv)
print (scores)


from scipy.stats import sem

#srednia + odchylenie
def mean_score(scores):
    """Print the empirical mean score and standard error of the mean."""
    return ("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), sem(scores))

print (mean_score(scores))