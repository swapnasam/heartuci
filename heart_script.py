# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:28:48 2020

@author: Swapna Sam
"""

import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

dataset = pd.read_csv("heart.csv")
length = len(dataset.columns)

#to check the correlation between variables and to remove higly correlated variables
cor = dataset.corr()
#couldnt find any highly correlated variables.

#to check missing values
msno.matrix(dataset)
#no missing values 


X = dataset.iloc[:,0 : length-2]
Y = dataset.iloc[:,length - 1]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

print(X_test.shape)

from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifiers = []
accuracy = 0
best_model = ""
arr = []

#to select best model based on accuracy

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
import xgboost

randClas = RandomForestClassifier(n_estimators = 1000, random_state = 42)
classifiers.append(randClas)
arr.append("RandomForestClassifier")

model1 = xgboost.XGBClassifier()
classifiers.append(model1)
arr.append("XGBClassifier")


model2 = svm.SVC()
classifiers.append(model2)
arr.append("svm")

logReg = LogisticRegression()
classifiers.append(logReg)
arr.append("LogisticRegression")

knn = KNeighborsClassifier()
classifiers.append(knn)
arr.append("LogistKNeighborsClassifiericRegression")

for i in range (0,len(classifiers)):
    classifiers[i].fit(X_train, y_train)
    y_pred= classifiers[i].predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_name = type(classifiers[i]).__name__
    print("%s" %(model_name))
    print("Accuracy is",acc)
    if(accuracy < acc):
        accuracy = acc
        best_model = arr[i]
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix  of %s is  \n%s " %(model_name, cm))
    
print("Best model with Accuracy %f is %s" %(accuracy, best_model))
