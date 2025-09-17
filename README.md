# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANJANA K L
RegisterNumber: 212224230241
*/
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

data.head:

<img width="1187" height="212" alt="image" src="https://github.com/user-attachments/assets/e5f097c1-dbc2-4d49-9829-9117dd07a34e" />

data1.head:

<img width="1180" height="221" alt="image" src="https://github.com/user-attachments/assets/c6c8a0e6-188a-4b95-a691-f422f987d8c4" />

data1.isnull().sum():

<img width="260" height="307" alt="image" src="https://github.com/user-attachments/assets/32659ff7-1dcf-48d4-895c-c591e610b381" />

data1.duplicated().sum():

<img width="62" height="31" alt="image" src="https://github.com/user-attachments/assets/4db0c70d-7f68-4780-8a03-9609eebde9e5" />


data1:

<img width="1000" height="203" alt="image" src="https://github.com/user-attachments/assets/1b99f75b-20c7-4134-bfcc-337c092afb38" />

x:

<img width="882" height="196" alt="image" src="https://github.com/user-attachments/assets/f0de6380-a5a2-4378-9db4-2409a2bedf69" />


y:

<img width="425" height="260" alt="image" src="https://github.com/user-attachments/assets/07aec19c-3e74-4fb4-bbaf-33587bafa6db" />

y_pred:

<img width="723" height="48" alt="image" src="https://github.com/user-attachments/assets/a393cd4f-f60c-41b1-9cf3-69d9eaafc7cc" />

Accuracy:

<img width="300" height="33" alt="image" src="https://github.com/user-attachments/assets/b15c062f-cd3b-4800-881e-39eead511d4d" />

Confusion:

<img width="122" height="68" alt="image" src="https://github.com/user-attachments/assets/10aecedf-c2ac-4bd8-9e17-d0657daa9767" />


Classification Report1:

<img width="580" height="195" alt="image" src="https://github.com/user-attachments/assets/4ab00f24-7333-481a-9b1c-e05cca72d5fd" />


<img width="1258" height="121" alt="image" src="https://github.com/user-attachments/assets/0dfbd73b-bcd6-4e0f-ace4-8ff3fbb31ae9" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
