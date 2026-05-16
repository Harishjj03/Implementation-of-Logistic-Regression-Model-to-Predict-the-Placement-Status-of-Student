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
# Logistic Regression for Student Placement Prediction

# 1️⃣ Import Libraries
import pandas as pd  
import numpy as np   
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns 

# 2️⃣ Load Dataset
data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head()) # display the dataset

# 3️⃣ Drop Unnecessary Columns
data = data.drop(["sl_no", "salary"], axis=1) # 1 = column 0 = row

# 4️⃣ Convert Target Variable (status) to Binary
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0}) # change status columns values with 0 & 1

# 5️⃣ Separate Features and Target
X = data.drop("status", axis=1)
y = data["status"]

# 6️⃣ One-Hot Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)  

print("\nAfter Encoding:")
print(X.head())

# 7️⃣ Feature Scaling   
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️⃣ Train-Test Split # split data for test + train
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9️⃣ Train Logistic Regression Model 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔟 Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # 

# 1️⃣1️⃣ Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()

#  SIGMOID FUNCTION GRAPH
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate values
x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

# Plot Sigmoid Curve
plt.figure(figsize=(6,4))
plt.plot(x_values, y_values)
plt.axhline(y=0.5, linestyle='--')
plt.axvline(x=0, linestyle='--')
plt.xlabel("Input Value")
plt.ylabel("Sigmoid Output (Probability)")
plt.title("Sigmoid Function Curve")
plt.grid(True)
plt.show()


```

## Output:
<img width="1031" height="923" alt="image" src="https://github.com/user-attachments/assets/389afcb6-cd33-4cc1-9558-bc3e13e37405" />
<img width="842" height="942" alt="image" src="https://github.com/user-attachments/assets/2326a8db-c524-40cf-bb8c-0d0514d281da" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
