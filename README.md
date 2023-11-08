# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
## Program:
```PYTHON
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GANESH R
RegisterNumber: 212222240029 
*/

import pandas as pd
data=pd.read_csv("dataset/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Data Head:
![ML1](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/4661c51a-0b5c-4d04-8373-e5d92a8ddf9c)

## Data set info:
![ML2](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/10200b2e-9519-437a-bf2e-65125885e9bb)

## Null dataset:
![ML3](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/fbc9458c-e3c2-4f93-9224-878ea4567d6b)

## Values count in left column:
![ML4](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/d0a7c246-ed01-43e9-8593-ea6cfd0a1023)

## Dataset transformed head:
![ML5](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/4cb7dcfa-2351-45ef-9113-acfc5e83c3f6)

## x.head:

![ML6](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/f05c5715-d605-4697-be91-34814600191e)

## Accuracy:

![ML6](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/4c4f0109-3a9a-403c-af83-558f8435876f)

## Data Prediction:

![ML8](https://github.com/ganesha360/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120884552/9ca17776-c61b-46d3-b72d-67bedf4df394)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
