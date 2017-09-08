# -*- coding: utf-8 -*-
##
"""
@author: grzegorz 2.0
"""
## imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import tempfile

## wczytanie danych do jednego zbioru, ponieważ przekształcenia takie same
data = pd.concat([pd.read_csv('titanic_train.csv'), pd.read_csv('titanic_test.csv')])

##Przetwarzanie danych wejściowych
data['FamilySize'] = data['SibSp'] + data['Parch']

data['Title'] = np.where(data.Name.str.contains('Mr\\.'), 'Mr', None)
data['Title'] = np.where(data.Name.str.contains('Mrs\\.'), 'Mrs', data['Title'])
data['Title'] = np.where(data.Name.str.contains('Master\\.'), 'Master', data['Title'])
data['Title'] = np.where(data.Name.str.contains('Miss\\.'), 'Miss', data['Title'])
data['Title'] = np.where(data.Name.str.contains(', Rev'), 'Rev', data['Title'])

data['Title'] = np.where(np.logical_and(data.Sex == 'male', pd.isnull(data['Title'])), 'Mr', data['Title'])
data['Title'] = np.where(pd.isnull(data.Title), 'Mrs', data['Title'])

data['Floor'] = data.Cabin.str[:1]
# data['Floor'] = np.where(pd.isnull(data.Floor), None, data['Floor'])

data['Embarked'] = np.where(pd.isnull(data.Embarked), 'S', data['Embarked'])

data['Fare'] = np.where(pd.isnull(data.Fare), 0, data['Fare'])


##LabelEncoder
#TODO: można przetestować oneHotEncoder
enc = LabelEncoder()
label_encoder = enc.fit(data['Title'])
print ("Categorical classes:", label_encoder.classes_)
integer_classes = label_encoder.transform(label_encoder.classes_)
print ("Integer classes:", integer_classes)
transformed = label_encoder.transform(data['Title'])
data['Title'] = transformed

enc = LabelEncoder()
label_encoder = enc.fit(data['Sex'])
data['Sex'] = label_encoder.transform(data['Sex'])

enc = LabelEncoder()
label_encoder = enc.fit(data[pd.notnull(data['Floor'])]['Floor'].values)
transformed = label_encoder.transform(data[pd.notnull(data['Floor'])]['Floor'].values)
indexes = pd.notnull(data.Floor)
data.loc[indexes, 'Floor'] = transformed

enc = LabelEncoder()
label_encoder = enc.fit(data['Embarked'])
data['Embarked'] = label_encoder.transform(data['Embarked'])

## predykcja wieku
# TODO: zobaczyć predykcję również tylko po Title
regresor = DecisionTreeRegressor();
X_train_age = data[pd.notnull(data.Age)][['Title', 'SibSp', 'Parch']]
y_train_age = data[pd.notnull(data.Age)][['Age']]
regresor.fit(X_train_age, y_train_age)

# TODO: sprawdzić tą predykcję wieku, działa chyba ok
# data['AgePredicted'] = np.where(pd.isnull(data.Age), regresor.predict(data[['Title', 'SibSp', 'Parch']]), None)
data['Age'] = np.where(pd.isnull(data.Age), regresor.predict(data[['Title', 'SibSp', 'Parch']]), data['Age'])


##predykcja poziomu
classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2)
#

X_train_floor = data[pd.notnull(data.Floor)][['Embarked', 'Pclass']]
y_train_floor = data[pd.notnull(data.Floor)]['Floor'].values.astype('int')
classifier.fit(X_train_floor, y_train_floor)


data['Floor'] = np.where(pd.isnull(data.Floor), classifier.predict(data[['Embarked', 'Pclass']]), data['Floor'])

##zmiana ceny za bilet
data['TicketCounts'] = data.groupby('Ticket')['Ticket'].transform('count')

data['Fare'] = data['Fare'] / data['TicketCounts']

##usunięcie nieużywanych kolumn
data = data.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch', 'TicketCounts'], axis = 1)

##zalozenie indeksu na kolumnie
data = data.set_index('PassengerId')

##Podzielenie na zbiór uczący i testowy
X_train = data[pd.notnull(data['Survived'])]
y_train = data[pd.notnull(data['Survived'])]['Survived'].values
y_train_tf = data[pd.notnull(data['Survived'])]['Survived']
X_train = X_train.drop(['Survived'], axis = 1)

X_test = data[pd.isnull(data['Survived'])]
X_test = X_test.drop(['Survived'], axis = 1)

##definicja metody uczącej i zapisującej dane
def learn_and_save(classifier, X_train, y_train, X_test, parameters, file_name):
    gs = GridSearchCV(
        classifier,
        parameters,
        verbose=2,
        n_jobs=3
    )

    gs.fit(X_train, y_train)
    print (file_name, ": ", gs.best_params_, gs.best_score_)
    pd.DataFrame(
        {'Survived':gs.predict(X_test)},
        index = X_test.index) \
        .to_csv(file_name, float_format = '%.0f')

##uczenie drzewo decyzyjne
classifier = DecisionTreeClassifier()
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.linspace(1, 10, 10).tolist() + [None],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 7, 9, 10]
}

learn_and_save(classifier, X_train, y_train, X_test, parameters, 'dtc_predict.csv')


##uczenie svc
classifier = SVC()
parameters = {
    'gamma': np.logspace(-2, 1, 4),
    'C': np.logspace(-1, 1, 3),
}
learn_and_save(classifier, X_train, y_train, X_test, parameters, 'svc_predict.csv')

## uczenie sąsiadami
classifier = KNeighborsClassifier()
# print (list(map(int, np.linspace(1, 20, 20).tolist())))
parameters = {
    'n_neighbors': list(map(int, np.linspace(1, 20, 20).tolist()))
}
learn_and_save(classifier, X_train, y_train, X_test, parameters, 'knn_predict.csv')

## Random forest classifier
classifier = RandomForestClassifier()
parameters = {
    'n_estimators': list(map(int, np.linspace(1, 20, 20).tolist())),
    'random_state': [10, 20, 33, 50, None],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

learn_and_save(classifier, X_train, y_train, X_test, parameters, 'rfc_predict.csv')
# X_train['Embarked'] = X_train['Embarked'].values.astype('str')



##Tensorflow DNN
age = tf.feature_column.numeric_column("Age")
fare = tf.feature_column.numeric_column("Fare")
family_size = tf.feature_column.numeric_column("FamilySize")

embarked = tf.feature_column.categorical_column_with_hash_bucket("Embarked", hash_bucket_size=1000)
floor = tf.feature_column.categorical_column_with_hash_bucket("Floor", hash_bucket_size=1000)
title = tf.feature_column.categorical_column_with_hash_bucket("Title", hash_bucket_size=1000)
sex = tf.feature_column.categorical_column_with_hash_bucket("Sex", hash_bucket_size=1000)
pclass = tf.feature_column.categorical_column_with_hash_bucket("Pclass", hash_bucket_size=1000)

columns_to_learn = [
    age,
    # fare,
    # family_size,
    # tf.feature_column.embedding_column(embarked, dimension=8),
    # tf.feature_column.indicator_column(floor),
    # tf.feature_column.indicator_column(title),
    # tf.feature_column.indicator_column(sex),
    # tf.feature_column.indicator_column(pclass),
]

train_steps = 2000

model_dir = tempfile.mkdtemp()


input_fn_titanic =  tf.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train_tf,
    batch_size=100,
    num_epochs=None,
    shuffle=True,
    num_threads=5)


m = tf.estimator.DNNClassifier(
    model_dir=model_dir,
    feature_columns=columns_to_learn,
    hidden_units=[100, 50])

m.train(
    input_fn=input_fn_titanic,
    steps=train_steps)