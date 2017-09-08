##import
import tempfile


import pandas as pd
import tensorflow as tf

##load test and train data
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"]

# metoda ladujaca dane
def input_fn(data_file, num_epochs, shuffle):
    """Input builder function."""
    df_data = pd.read_csv(
        tf.gfile.Open(data_file),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    #zamiana na 0, 1
    labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
    return tf.estimator.inputs.pandas_input_fn(
        x=df_data,
        y=labels,
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=5)


## zamiana kolumn na liczbowe
# jeżeli wiemy ile mamy kategorii

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

# Jeżeli nie wiemy ile mamy kategorii
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

#ustawienie, że wartości są numeryczne
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")


##inne operacje na kolumnach
#podział na zbiory
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

#laczenie pol (crossing)
education_x_occupation = tf.feature_column.crossed_column(["education", "occupation"], hash_bucket_size=1000)
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column([age_buckets, "education", "occupation"], hash_bucket_size=1000)
native_country_x_occupatuion = tf.feature_column.crossed_column(["native_country", "occupation"], hash_bucket_size=1000)

##definicja pol
base_columns = [
    gender, native_country, education, occupation, workclass, relationship, age_buckets,
]
crossed_columns = [education_x_occupation, age_buckets_x_education_x_occupation, native_country_x_occupatuion]

print (tf.feature_column.indicator_column(workclass))
print (workclass)
deep_columns = [
    # Use indicator columns for low dimensional vocabularies
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(relationship),
    # To show an example of embedding
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
]

##Metoda zwracająca estymatory
def build_estimator(model_dir, model_type):
    """Build an estimator."""
    if model_type == "wide":
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=base_columns + crossed_columns)
    elif model_type == "deep":
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50])
    else:
        m = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=crossed_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50])
    return m

##Metoda ucząca i sprawdzająca
def train_and_eval(model_dir, model_type, train_steps):
    """Train and evaluate the model."""
    train_file_name, test_file_name = "adult.data", "adult.test"
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir

    m = build_estimator(model_dir, model_type)
    # set num_epochs to None to get infinite stream of data.
    m.train(
        input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
        steps=train_steps)
    # set steps to None to run evaluation until all data consumed.
    results = m.evaluate(
        input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
        steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


## Uczenie liniowe
model_type = "wide"
train_steps = 2000
model_dir = ""


train_and_eval(model_dir, model_type, train_steps)

## Uczenie DNN
model_type = "deep"
train_steps = 2000
model_dir = ""


train_and_eval(model_dir, model_type, train_steps)