# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Prashanth.K
RegisterNumber:  212223230152
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```
## Output:
# Dataset:
![image](https://github.com/user-attachments/assets/6e6dc080-7226-4936-8378-60e040fc80b4)

# Head values:
![image](https://github.com/user-attachments/assets/12c23a49-1461-412a-892c-3c91ad48de2b)

# Tail values:
![image](https://github.com/user-attachments/assets/1e9219e7-35ba-4cc9-bdb9-ccb48f5fe3e5)

# X and Y values:
![image](https://github.com/user-attachments/assets/cf395c2d-3466-4c5c-a27f-3ca295d88b22)

# Predication values of X and Y:
![image](https://github.com/user-attachments/assets/2de3f2fb-9abf-46a4-9791-1f9abd563414)

# MSE,MAE and RMSE:
![image](https://github.com/user-attachments/assets/71209a17-c830-4b57-ac78-850639472516)

# Training Set:
![image](https://github.com/user-attachments/assets/7ddf31cf-e875-4518-9e8d-d98956a6f99f)

# Testing Set:
![image](https://github.com/user-attachments/assets/ca5051ed-35d4-4586-a2c3-82154eee47e9)













## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
