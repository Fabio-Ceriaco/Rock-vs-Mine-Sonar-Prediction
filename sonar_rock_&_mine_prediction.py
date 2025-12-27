# WorkFlow
# 1. Sonar Data
# 2. Data Preprocessing
# 3. Train Test Split
# 4. Model Training (Logistic Regression) Logistic Regression is used for binary classification problems, where the goal is to predict one of two possible outcomes based on input features.
# 5. Model Evaluation
# 6. New Data Prediction
# 7. Model Evaluation


# Importing the Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)  # For splitting the dataset into training and testing sets
from sklearn.linear_model import LogisticRegression  # For Logistic Regression model

from sklearn.metrics import accuracy_score  # For evaluating the model's accuracy

pd.set_option("display.max_columns", 61)  # To display all columns of the dataframe
# 1. Sonar Data

# Loading the dataset

# header=None to indicate that the CSV file does not have a header row that way pandas will automatically assign integer column names starting from 0
sonar_data = pd.read_csv("sonar data.csv", header=None)

# print(sonar_data.head())  # Display the first few rows of the dataset

# number of rows and columns in the dataset
# print(sonar_data.shape)

# 2. Data Preprocessing

# Counting the number of instances for each class label
# there are two class labels in the dataset: 'R' (Rock) and 'M' (Mine)
# there are 111 instances of 'M' and 97 instances of 'R'
# to see the distribution of class labels in the dataset
# we can use the value_counts() function on the last column (column index 60)
# this dataset is balanced as both classes have a similar number of instances
# print(sonar_data[60].value_counts())

# Looking for missing values
# print(sonar_data.isnull().sum())

# Descriptive statistics of the dataset
# describe gives statistical summary of the dataset
# print(sonar_data.describe())

# Separating features and labels
X = sonar_data.drop(columns=60, axis=1)  # Features (all columns except the last one)
y = sonar_data[60]  # Labels (the last column)

# print(X)
# print(y)

# Grouping the data based on labels to see the mean values of features for each class

# print(sonar_data.groupby(60).mean())
# Conclusion: The mean values of the features differ between the two classes ('R' and 'M'), indicating that the features can help distinguish between rocks and mines.


# 3. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=2
)
# test_size=0.1 - means 10% of the data will be used for testing and 90% for training
# stratify=y - ensures that the class distribution in the training and testing sets is similar
# to that of the original dataset to we have balanced classes in both sets same number of 'R' and 'M' instances.
# random_state=2 - is used to ensure reproducibility of the results

# print(f"X.shape: {X.shape}")
# print(f"X_train.shape: {X_train.shape}")
# print(f"X_test.shape: {X_test.shape}")
# print(f"y.shape: {y.shape}")
# print(f"y_train.shape: {y_train.shape}")
# print(f"y_test.shape: {y_test.shape}")


# 4. Model Training (Logistic Regression)

# print(X_train)
# print(y_train)

model = LogisticRegression()

# training the Logistic Regression model with the training data
model.fit(X_train, y_train)

# 5. Model Evaluation

# accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print(f"Accuracy on Training data: {training_data_accuracy}")

# accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print(f"Accuracy on Test data: {test_data_accuracy}")
# Conclusion: The Logistic Regression model performs well on both the training and test datasets, indicating that it has learned to distinguish between rocks and mines effectively.


# Making a Predictive System

input_data = (
    0.0283,
    0.0599,
    0.0656,
    0.0229,
    0.0839,
    0.1673,
    0.1154,
    0.1098,
    0.1370,
    0.1767,
    0.1995,
    0.2869,
    0.3275,
    0.3769,
    0.4169,
    0.5036,
    0.6180,
    0.8025,
    0.9333,
    0.9399,
    0.9275,
    0.9450,
    0.8328,
    0.7773,
    0.7007,
    0.6154,
    0.5810,
    0.4454,
    0.3707,
    0.2891,
    0.2185,
    0.1711,
    0.3578,
    0.3947,
    0.2867,
    0.2401,
    0.3619,
    0.3314,
    0.3763,
    0.4767,
    0.4059,
    0.3661,
    0.2320,
    0.1450,
    0.1017,
    0.1111,
    0.0655,
    0.0271,
    0.0244,
    0.0179,
    0.0109,
    0.0147,
    0.0170,
    0.0158,
    0.0046,
    0.0073,
    0.0054,
    0.0033,
    0.0045,
    0.0079,
)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

print(len(input_data_as_numpy_array))
# reshaping the array as we are predicting for one instance
# reshaping to 2D array with 1 row and appropriate number of columns
# we need to reshape because the model expects a 2D array as input
# the -1 in reshape means that the number of columns will be inferred based on the length of the input_data.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prediction
for input_data in input_data_reshaped:
    prediction = model.predict(input_data_reshaped)

    # print(f"Prediction: {prediction}")

    if prediction[0] == "R":
        print("The object is a Rock")
    else:
        print("The object is a Mine")
    # Conclusion: The predictive system uses the trained Logistic Regression model to classify new sonar data as either a rock or a mine based on the input features provided.
