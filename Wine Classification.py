# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 20:28:32 2021
WINE DATASET REGRESSION
@author: Dennoh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

wine = pd.read_csv('/Users/USER/AppData/Local/Programs/Python/Python39/winequality-red.csv')

#Exploratory Data Analysis
wine.shape

wine.head()
wine.isnull().sum()
wine.describe()
wine.info()
wine.duplicated().sum()

#VISUALIZATION
from autoviz.AutoViz_Class import AutoViz_Class
AV=AutoViz_Class()
AV.AutoViz(filename='',depVar='quality',dfte=wine,verbose=0)

corr_wine = wine.corr()
print(corr_wine)
ax_corr = sns.heatmap(
    corr_wine,
    vmin=-1,vmax=1,center=0,
    cmap= 'viridis',
    square=True)

newwine=pd.DataFrame(wine.drop(columns=['quality']))
for wd in newwine:
    fig=plt.figure(figsize=(10,8))
    sns.barplot(x=wine['quality'],y=newwine[wd])
    
#DATA PREPROCESSING
X=newwine
X.head()
y=pd.factorize(wine['quality'])[0].reshape(-1,1)

y.shape

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaledX=scaler.fit_transform(X)
print(scaledX)




from sklearn.preprocessing import Binarizer
binarizer = Binarizer(7)
binarizer.fit_transform(y)

from sklearn.model_selection import train_test_split
scaledX_train,scaledX_test,y_train,y_test = train_test_split(scaledX,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(scaledX_train,y_train)
y_pred=regressor.predict(scaledX_test)
print(y_pred) 
regressor.intercept_

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred,y_test)
print(acc)
    

