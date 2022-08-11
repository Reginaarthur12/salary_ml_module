#importing libraries
import numpy as np
import pandas as pd
import matplotlib as mtp

#create variable to store data
salary_data=pd.read_csv('salary_Data.csv')

#ssigning x and y to dependent and independent variable
x = salary_data.iloc[:,0:1].values
y = salary_data.iloc[:,1:2].values
 
#splitting the dataset into train data and test data 
from sklearn.model_selection import train_test_split 

#creating variable to store x_train,x_test and y_train, y_test
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=1/3, random_state=0)

#feature scaling
from sklearn.linear_model import LinearRegression

#assing the LineRScaler to a variable salary_module
salary_module =LinearRegression()

#fitting and transforming the x train and the x test
salary_module.fit(x_train,y_train)

#fiftting x_test
salary_prediction =salary_module.predict(x_test)
