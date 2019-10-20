#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:32:26 2019

@author: Bharath
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

#Importing dataset 
dataset = pd.read_csv('SolarPrediction.csv')
dataset = dataset.sort_values(["UNIXTime"], ascending = False)
dataset = dataset.head(10000)
#dataset_preprocessed = dataset[dataset['Radiation']<200]
#dataset = dataset.drop(dataset_preprocessed.index,axis =0)
X = dataset.iloc[:,0:1].values
Y = dataset.iloc[:,4:5].values


start = timeit.default_timer()
#splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_Test = train_test_split(X,Y,test_size = 1/9, random_state = 0)
stop = timeit.default_timer()
print('TIme taken to train Data:', stop-start)

#Fitting sample Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set result
y_pred = regressor.predict(X_test)


#Visualizing the Training set result
plt.scatter(X_train,Y_train, color ='red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Time vs Temperature [Training set]')
plt.xlabel('Time in UnixTimeStamp')
plt.ylabel('Temperatue in Fahrenheit')
plt.show()

#Visualizing the Test set result
plt.scatter(X_test,Y_Test, color ='red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Time vs Temperature [Test set]')
plt.xlabel('Time in UnixTimeStamp')
plt.ylabel('Temperatue in Fahrenheit')
plt.show()