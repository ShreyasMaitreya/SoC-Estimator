#training data: https://mega.nz/folder/sBgRVATQ#fnreL6PprOyhsjtNw008xQ
import tkinter
from tkinter import filedialog
import os
import pandas as pd

root = tkinter.Tk()
root.withdraw() #use to hide tkinter window

currdir = os.getcwd()
df = pd.read_csv (r"D:\Battery\Actual\LNO\30degC LNO.csv") #import training data

from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.impute import SimpleImputer

imputer= SimpleImputer(fill_value=np.nan,strategy='mean')
data = imputer.fit_transform(df)
x = data[:,[0,1,2]]
y = data[:,1]

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=7)
poly_variables = poly.fit_transform(x)

poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, y, test_size = 0.2, random_state = 4)

regression = linear_model.LinearRegression()

model = regression.fit(poly_var_train, res_train)
score = model.score(poly_var_test, res_test)

y_pred =  abs(regression.predict(poly_variables))
from sklearn.metrics import r2_score, mean_squared_error,explained_variance_score,mean_absolute_error
print("R2 Score:" + str(r2_score(y,y_pred)))
print("Variance Score:" + str(explained_variance_score(y,y_pred)))
print("Mean Squared Error:" + str(mean_squared_error(y,y_pred)))
print("Mean Absolute Error:" + str(mean_absolute_error(y,y_pred)))

err = abs(y_pred-y)
df = pd.DataFrame(err)
df.to_csv('err30degLNO.csv') #keep updating the filename according to the file you have imported

