from pprint import pprint

from sklearn import datasets

iris = datasets.load_iris()
X_iris, Y_iris = iris.data, iris.target

pprint(iris)

print (X_iris.shape, Y_iris.shape)

print(X_iris[0], Y_iris[0])