#sprawdza z jakiej listy dyskusyjnej jest dany mail

##
import pprint
import numpy as np

## load data
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')

##partition data to training and testing set
SPLIT_PERC = 0.75
split_size = int(len(news.data)*SPLIT_PERC)
X_train = news.data[:split_size]
X_test = news.data[split_size:]
y_train = news.target[:split_size]
y_test = news.target[split_size:]

##cross validation
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), sem(scores)))

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer

clf_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])
clf_2 = Pipeline([
    ('vect', HashingVectorizer(non_negative=True)),
    ('clf', MultinomialNB()),
])
clf_3 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

clf_4 = Pipeline([
    ('vect', TfidfVectorizer(
         token_pattern='[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+'
        # token_pattern='\b[a-z]+\b'
    )),
    ('clf', MultinomialNB()),
])
clfs = [clf_1, clf_2, clf_3]
for clf in clfs:
    evaluate_cross_validation(clf, news.data, news.target, 5)

evaluate_cross_validation(clf_4, news.data, news.target, 5)

##test
tfid = TfidfVectorizer(
    # token_pattern='\b[a-z]+\b'
    # token_pattern='[a-z]+'
    token_pattern='[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+'
)
tfid.fit_transform(["jakis tekst napisany tutaj jest tutaj jest bo tak jest"])
xxx = tfid.get_feature_names();
# xxx = tfid.transform(["jakis tekst napisany ttaj"])
print(xxx)

##
# pobieramy stopwordsy
def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', 'r').readlines():
        result.add(line.strip())
    return result

stop_words = get_stop_words()

clf_5 = Pipeline([
    ('vect', TfidfVectorizer(
        token_pattern='[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+',
        stop_words= stop_words
    )),
    ('clf', MultinomialNB()),
])

evaluate_cross_validation(clf_5, news.data, news.target, 5)

## bawimy się MutinominalNB

clf_7 = Pipeline([
    ('vect', TfidfVectorizer(
        token_pattern='[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+',
        stop_words= stop_words
    )),
    ('clf', MultinomialNB(alpha=0.01)),
])

evaluate_cross_validation(clf_7, news.data, news.target, 5)

##
from sklearn import metrics

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(clf_7, X_train, X_test, y_train, y_test)

print (len(clf_7.named_steps['vect'].get_feature_names()))

feature_names = clf_7.named_steps['vect'].get_feature_names()