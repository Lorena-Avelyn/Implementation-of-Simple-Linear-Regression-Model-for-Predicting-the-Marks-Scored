# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

  1.Import the standard Libraries.
  
  2.Set variables for assigning dataset values.
  
  3.Import linear regression from sklearn.
  
  4.Assign the points for representing in the graph.
  
  5.Predict the regression for marks by using the representation of the graph.
  
  6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
df= pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
y_test
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```
Developed by: Lorena Avelyn R

Register Number: 212224040174

## Output:

<img width="1271" height="627" alt="Screenshot 2025-08-21 142837" src="https://github.com/user-attachments/assets/08362b45-5652-4f5c-83f4-8b398dbe42f2" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
