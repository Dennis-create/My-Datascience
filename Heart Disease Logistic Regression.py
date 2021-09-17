# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:21:46 2021

@author: USER
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Import the data

data=pd.read_csv('/Users/USER/AppData/Local/Programs/Python/Python39/framingham_heart_disease.csv')

#Data Wrangling
data.head
data.info()
data.shape

#Checking null values
data.isnull().sum()

#Drop education since it's irrelevant
data.drop(['education'],axis=1,inplace=True)
data.shape

#Replace null values with mean
x=data['cigsPerDay'].mean()
data['cigsPerDay'].fillna(x,inplace=True)
y=data['BPMeds'].mean()
data['BPMeds'].fillna(y,inplace=True)
z=data['totChol'].mean()
data['totChol'].fillna(z,inplace=True)
a=data['BMI'].mean()
data['BMI'].fillna(a,inplace=True)
b=data['heartRate'].mean()
data['heartRate'].fillna(b,inplace=True)
c=data['glucose'].mean()
data['glucose'].fillna(c,inplace=True)

#Data Visualization

from autoviz.AutoViz_Class import AutoViz_Class
AV=AutoViz_Class()
dft=AV.AutoViz(filename='',depVar='TenYearCHD',dfte=data,verbose=0,lowess=False)

data.corr()

X=data[['age','cigsPerDay','BMI','totChol','sysBP','diaBP','glucose','heartRate',]]

y=data['TenYearCHD']

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaledX=scaler.fit_transform(X)
print(scaledX)


from sklearn.model_selection import train_test_split
scaledX_train,scaledX_test,y_train,y_test=train_test_split(scaledX,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(scaledX_train, y_train)
prediction_y = model.predict(scaledX_test)
model.coef_
model.intercept_

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,prediction_y)
print(accuracy)










