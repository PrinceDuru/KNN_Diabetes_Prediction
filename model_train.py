# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 02:47:33 2022

@author: Prince
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

#Load Data
file = "preprocessed_data.csv"
path = r"datasets"
filepath = os.path.join(path, file)

df = pd.read_csv(filepath)

df.head()

# Get top 5 features/columns
topFeatures = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies']


# Set the Predictor and Dependent variables
X = df[topFeatures]
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 40)

# Check shape of split data
X_train.shape
X_test.shape
y_train.shape
y_test.shape


# Standardize data using standard Scaler
scaler = StandardScaler()
X_train_STd = scaler.fit_transform(X_train)

# Transform input test data for prediction -- Test data will be transformed on the fly through POST request
# X_test_st = scaler.transform(X_test)


# Instantiate KNeighborsClassifier class and fit the model

KNN_model = KNeighborsClassifier(n_neighbors = 24)
KNN_model.fit(X_train_STd, y_train)


# Fit / Train the Model
KNN_model.fit(X_train_STd, y_train)

# Save Model as pickle file
pickle.dump(KNN_model, open('model.pkl', 'wb'))
