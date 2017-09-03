#algorytm k-means (k-średnich)

## wylaczenie deprecate warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

## ladowanie bibliotek
import pprint
import numpy as np
import matplotlib.pyplot as plt

## importowanie oraz wyświetlenie
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digits = load_digits()
data = scale(digits.data)

def print_digits(images,y,max_n=10):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    i = 0
    while i < max_n and i < images.shape[0]:
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        p.text(0, 14, str(y[i]))
        i = i + 1

print_digits(digits.images, digits.target, max_n=10)


## podział na dane uczące i testowe
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    data, digits.target, digits.images,  test_size=0.25, random_state=42)

n_samples, n_features = X_train.shape
n_digits = len(np.unique(y_train))
labels = y_train


print_digits(images_train, y_train, max_n=20)
print_digits(images_test, y_test, max_n=20)

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

## Uczenie k-means
from sklearn import cluster
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
clf.fit(X_train)
print (clf.labels_.shape)
print (clf.labels_[1:10])
print_digits(images_train, clf.labels_, max_n=10)

## predykcja
y_pred = clf.predict(X_test)

def print_cluster(images, y_pred, cluster_number):
    images = images[y_pred==cluster_number]
    y_pred = y_pred[y_pred==cluster_number]
    print_digits(images, y_pred, max_n=10)

for i in range(10):
    print_cluster(images_test, y_pred, i)

## evaluate
from sklearn import metrics
print ("Addjusted rand score:{:.2}".format(metrics.adjusted_rand_score(y_test, y_pred)))
print ("Homogeneity score:{:.2} ".format(metrics.homogeneity_score(y_test, y_pred)))
print ("Completeness score: {:.2} ".format(metrics.completeness_score(y_test, y_pred)))

# W matrycy kolumny, to predykcje - wiersze dane testowe
print ("Confusion matrix")
print (metrics.confusion_matrix(y_test, y_pred))

#alternatywne metody klastrowania

## AffinityPropagation - sam znajduje ilosc klastrów
aff = cluster.AffinityPropagation()
aff.fit(X_train)
#znalazlo tutaj 112 klastrów
print (aff.cluster_centers_indices_.shape)
y_pred2 = aff.predict(X_test)

## MeanShift
ms = cluster.MeanShift()
ms.fit(X_train)
print (ms.cluster_centers_.shape)

## Guassian Mixure Models
from sklearn import mixture

# Define a heldout dataset to estimate covariance type
X_train_heldout, X_test_heldout, y_train_heldout, y_test_heldout = train_test_split(
    X_train, y_train,test_size=0.25, random_state=42)
for covariance_type in ['spherical','tied','diag','full']:
    gm=mixture.GMM(n_components=n_digits, covariance_type=covariance_type, random_state=42, n_init=5)
    gm.fit(X_train_heldout)
    y_pred=gm.predict(X_test_heldout)
    print ("Adjusted rand score for covariance={}:{:.2}".format(covariance_type, metrics.adjusted_rand_score(y_test_heldout, y_pred)))

## wybór coveriance_type, bo tied wyszło najlepiej
gm = mixture.GMM(n_components=n_digits, covariance_type='tied', random_state=42)
gm.fit(X_train)

y_pred = gm.predict(X_test)
print ("Addjusted rand score:{:.2}".format(metrics.adjusted_rand_score(y_test, y_pred)))
print ("Homogeneity score:{:.2} ".format(metrics.homogeneity_score(y_test, y_pred)))
print ("Completeness score: {:.2} ".format(metrics.completeness_score(y_test, y_pred)))
for i in range(10):
    print_cluster(images_test, y_pred, i)
print ("Confusion matrix")
print (metrics.confusion_matrix(y_test, y_pred))