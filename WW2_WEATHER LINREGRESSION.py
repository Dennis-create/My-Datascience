# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ww2_weather = pd.read_csv('/Users/USER/AppData/Local/Programs/Python/Python39/Summary of Weather.csv')
weather=pd.DataFrame(ww2_weather)

weather.head() #Show first 5 
print(weather.head())

#DATA WRANGLING

weather.drop(['Precip','Date','STA','WindGustSpd','Snowfall','PoorWeather', 'PRCP', 'DR', 'YR','MO','DA','SPD', 'SNF', 'SND', 'FT', 'FB', 'FTI', 'ITH', 'PGT', 'TSHDSBRSGF', 'SD3', 'RHX', 'RHN', 'RVG', 'WTE'],inplace=True,axis=1)
weather.isnull().sum()
print(weather.shape)

#Replacing null values with the mean

x = weather['MAX'].mean()
weather['MAX'].fillna(x,inplace=True)
y = weather['MIN'].mean()
weather['MIN'].fillna(y,inplace=True)
z = weather['MEA'].mean()
weather['MEA'].fillna(z,inplace=True)
print(weather.head)

weather.info()
weather.describe()
weather.corr()

#DATA VISUALIZATION

sns.countplot(x='MaxTemp',data=weather,palette='magma')
plt.title('World War 2 Weather')
plt.show()

sns.regplot(x='MaxTemp',y='MinTemp',data=weather)

sns.heatmap(weather.corr(),cmap='viridis')
plt.xticks(rotation=-45) #rotate xlabels

sns.pointplot(x='MaxTemp', y='MinTemp',data=weather)
plt.show()

sns.countplot(x='MinTemp', data=weather)

sns.distplot(weather['MinTemp'],kde=False)

sns.relplot(x='MaxTemp',y='MinTemp',data=weather,kind='line',ci=None)
sns.distplot(weather['MaxTemp'])


#Building the linear regression model

X=weather['MinTemp']
y=weather['MaxTemp']

X=weather.iloc[:,:1].values
y=weather.iloc[:,2].values
print(X)
print(y)


from sklearn.model_selection import train_test_split
(X_train,X_test,y_train,y_test)=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#Prediction model

y_pred=regressor.predict(X_test)
print(y_pred)

#Test accuracy
from sklearn.metrics import r2_score
z=r2_score(y_pred, y_test)
print(z)

print(regressor.coef_)



               

               







