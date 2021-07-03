# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:58:18 2020

@author: USER
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df_cars = pd.read_csv('auto-mpg.csv')

df_cars = df_cars[df_cars.horsepower != '?']

df_cars.horsepower = df_cars.horsepower.astype('float')

x = df_cars.drop('mpg', axis=1)
y = df_cars['mpg']


df_cars=df_cars.drop('car name',axis=1)

x = df_cars.drop(['mpg'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.25)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)

    
pickle.dump(lr,open('model.pkl','wb'))

