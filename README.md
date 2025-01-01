# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Problem Definition

Identify the independent variables (X) such as experience, education_level, and performance_rating.
The dependent variable (Y) is the salary of the employee.

2.Load and Preprocess Data

Load the dataset, handle missing values, and encode categorical variables if necessary.

3.Split Data

Divide the dataset into training and testing subsets.

4.Initialize Decision Tree Regressor

Use DecisionTreeRegressor from sklearn. Set parameters like max_depth to prevent overfitting.

5.Train the Model

Fit the regressor on the training data.

6.Make Predictions

Predict employee salaries on the test data.

7.Evaluate the Model

Use metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score to evaluate the model.

7.Visualize Decision Tree (Optional)

Plot the decision tree to interpret feature splits and thresholds.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: vishal.v
RegisterNumber: 24900179 
*/
import pandas as pd
 from sklearn.tree import DecisionTreeClassifier,plot_tree
 from sklearn.preprocessing import LabelEncoder
 data=pd.read_csv(r"C:\Users\admin\Downloads\Employee.csv")
 print(data.head())
 print(data.info())
 print(data.isnull().sum())
 data["left"].value_counts()
 
 le=LabelEncoder()
 data["salary"]=le.fit_transform(data["salary"])
#  print(data.head())
 x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
#  print(x.head())    
 y=data["left"]
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
 from sklearn.tree import DecisionTreeClassifier
 dt=DecisionTreeClassifier(criterion="entropy")
 dt.fit(x_train,y_train)
 y_pred=dt.predict(x_test)
 from sklearn import metrics
 accuracy=metrics.accuracy_score(y_test,y_pred)
#  print(accuracy)
 dt.predict([[0.5,0.8,9,260,6,0,1,2]])
 import matplotlib.pyplot as plt
 plt.figure(figsize=(8,6))
 plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
 plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/49065fdb-c929-4a90-87e5-012633422bbf)
![image](https://github.com/user-attachments/assets/fada14ae-f990-43d3-b708-6ae2ae188096)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
