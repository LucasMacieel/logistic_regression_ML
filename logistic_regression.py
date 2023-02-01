# Libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a dataframe named ad_data using pandas based on a csv file
ad_data = pd.read_csv('advertising.csv')
'''
   Daily Time Spent on Site  Age  Area Income  Daily Internet Usage  ... Male     Country            Timestamp Clicked on Ad
0                     68.95   35     61833.90                256.09  ...    0     Tunisia  2016-03-27 00:53:11             0
1                     80.23   31     68441.85                193.77  ...    1       Nauru  2016-04-04 01:39:02             0
2                     69.47   26     59785.94                236.50  ...    0  San Marino  2016-03-13 20:35:42             0
3                     74.15   29     54806.18                245.89  ...    1       Italy  2016-01-10 02:31:19             0
4                     68.37   35     73889.99                225.58  ...    0     Iceland  2016-06-03 03:36:18             0
'''
# Importing train_test_split function to create training and testing datasets 
from sklearn.model_selection import train_test_split

# Selecting the columns of the dataframe which will be used to predict y
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]

# Data regarding if user clicked or not on ad (0 = did not click, 1 = clicked)
y = ad_data['Clicked on Ad']

# Spliting datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Training and fitting a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Predicting values for the testing data
predictions = logmodel.predict(X_test)

# Creating a classification report and confusion matrix for the model

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

'''
              precision    recall  f1-score   support

           0       0.86      0.96      0.91       162
           1       0.96      0.85      0.90       168

    accuracy                           0.91       330
   macro avg       0.91      0.91      0.91       330
weighted avg       0.91      0.91      0.91       330
'''

# Classification report showed that the model had an accuracy of 91%

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predictions))
'''
[[True Positives   False Negatives]
 [False Positives True Negatives]]

[[156   6]
 [ 25 143]]

 Accuracy = (TP + TN)/Total -> Accuracy = (156 + 143)/330 = 0.906 = 91%
'''