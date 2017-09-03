##importy
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt


## pobranie danych
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
n_samples = 3000 #ograniczamy liczbę
X_train = news.data[:n_samples]
y_train = news.target[:n_samples]

## import stop words
def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', 'r').readlines():
        result.add(line.strip())
    return result

stop_words = get_stop_words()

## crate pipeline
clf = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words=stop_words,
        token_pattern='[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+'
    )),
    ('nb', MultinomialNB(alpha=0.01)),
])

##cross validation
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print (("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

evaluate_cross_validation(clf, X_train, y_train, 3)

## calc_params

def calc_params(X, y, clf, param_values, param_name, K):
    # initialize training and testing scores with zeros
    train_scores = np.zeros(len(param_values))
    test_scores = np.zeros(len(param_values))

    # iterate over the different parameter values
    for i, param_value in enumerate(param_values):
        print (param_name, ' = ', param_value)

        # set classifier parameters
        clf.set_params(**{param_name:param_value})

        # create KFold cross validation
        cv = KFold(K, shuffle=True, random_state=0)

        scores = cross_val_score(clf, X, y, cv=cv)
        print (scores)

        # store the mean of the K fold scores
        train_scores[i] = np.mean(scores)

    # plot the training and testing scores in a log scale
    plt.semilogx(param_values, train_scores, alpha=0.4, lw=2, c='b')
    # plt.semilogx(param_values, test_scores, alpha=0.4, lw=2, c='g')

    plt.xlabel(param_name + " values")
    plt.ylabel("Mean cross validation accuracy")

    # return the training and testing scores on each parameter value
    return train_scores

## list of alfha values
alphas = np.logspace(-7, 0, 8)
print (alphas)

##
train_scores = calc_params(X_train, y_train, clf, alphas, 'nb__alpha', 3)
#najlepszy score jest dla alpha = 0.1
print (train_scores)

## kolejny pipeline do testowania parametrów
from sklearn.svm import SVC

clf_gamma = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words=stop_words,
        token_pattern='[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+'
    )),
    ('svc', SVC()),
])

## wygenerowanie gamma i sprawdzenie
gammas = np.logspace(-2, 1, 4)

train_scores = calc_params(X_train, y_train, clf_gamma, gammas, 'svc__gamma', 3)

print (train_scores)
#tutaj wyszło najlepsze gamma 1


## Grid search poszukiwanie kombinacji parametrów
from sklearn.model_selection import GridSearchCV

parameters = {
    'svc__gamma': np.logspace(-2, 1, 4),
    'svc__C': np.logspace(-1, 1, 3),
}

clf_grid = Pipeline([
    ('vect', TfidfVectorizer(
        stop_words=stop_words,
        token_pattern='[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+'
    )),
    ('svc', SVC()),
])

gs = GridSearchCV(clf_grid, parameters, verbose=2, refit=False, cv=3)

gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_
#wyszło z tego, że najlepsze to gamma: 0.1, c: 10
#Zajęło to ponad 8 minut
## parallel grid search
# wystarczy dodać parametr n_jobs (-1 - maks corów)
gs = GridSearchCV(clf_grid, parameters, verbose=2, refit=False, cv=3, n_jobs=-1)

gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_

#to samo co wyżej tylko zrobiło w 2.5 minuty