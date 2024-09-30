# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ZAFREEN J
RegisterNumber:  212223040252
*/
```

```
import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

![image](https://github.com/user-attachments/assets/737885ff-b71b-402a-b255-e4f5671d4f5d)

![image](https://github.com/user-attachments/assets/1b28d8e8-6cf7-4111-986b-62810d389a68)

![image](https://github.com/user-attachments/assets/59ae43d8-4bf0-4353-9497-fc3bf82b4199)

![image](https://github.com/user-attachments/assets/91062ca9-5cd7-4663-a788-5a44213d2318)

![image](https://github.com/user-attachments/assets/7b5f174a-e08a-4bcf-bc6e-b2add6f32be6)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
