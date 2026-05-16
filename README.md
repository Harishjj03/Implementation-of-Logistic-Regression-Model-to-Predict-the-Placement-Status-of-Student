# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables.
2.Define the features (X) and target variable (y). 
3.Split the data into training and testing sets.
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HARISHBALA J
RegisterNumber: 212224223002 
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load data
placement = pd.read_csv("Placement_Data.csv")

# Input columns
X = placement[['ssc_p', 'hsc_p', 'mba_p']]

# Output column
Y = placement['status']

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0
)

# Create model
classifier = LogisticRegression()

# Fit model
classifier.fit(x_train, y_train)

# Predict
y_pred = classifier.predict(x_test)

# Print predictions
print("Predicted Values:")
print(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Accuracy
accuracy = classifier.score(x_test, y_test)

print("\nAccuracy :", accuracy)
```

## Output:
<img width="1175" height="340" alt="image" src="https://github.com/user-attachments/assets/9a63a07c-ca41-4770-a2bc-fd1bb5a1d89c" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
