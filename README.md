## Linear Regression: Predicting Student Final Scores

This project demonstrates the application of **Linear Regression** to predict a student's final exam score based on various demographic features.

### Problem Statement

   Your task is to build a model that predicts students 'final exams score based on their study routines. Create a Google Form to collect      data on:
  
    Daily hours of sleep
    Use of mobile while studying
    Sleep hours before exam
    Attendance percentage
    Participation in doubt sessions

Use this data to build a linear regression model to predict the final score out of 100.Focus on data preprocessing, handling missing values, and model evaluation using RMSE or MAE

### Dataset

The dataset used in this project contains student information, including:

- `Hours_Studied`
- `Daily hours of sleep`
- `Use of mobile while studying`
- `Attendance_Percentage`
- `Participation`
- `Final_Score` (target variable)

### Approach

1. **Data Preparation:**  
   - Load the dataset and inspect for missing values or outliers.
   - Perform necessary data cleaning and preprocessing.

2. **Model Building:**  
   - Use scikit-learn’s `LinearRegression` model.
   - Fit the model on the training data.
   - Predict final scores on the test data.

3. **Evaluation:**  
   - Evaluate the model using metrics such as Mean Squared Error (MSE)
   - Visualize predictions vs. actual scores.

### Example Code

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
sc=pd.read_csv('/content/Final Exam Score (1).csv')
sc.head()
sc.isnull().sum()
sc.dropna(inplace=True)
sc.info()
from sklearn.preprocessing import LabelEncoder
x=LabelEncoder()
sc['Enter_Your_Name']=x.fit_transform(sc['Enter Your Name'])
x1=LabelEncoder()
sc['Usage of mobile phone while studying']=x.fit_transform(sc['Usage of mobile phone while studying'])
x2=LabelEncoder()
sc['Participation in doubt sessions']=x.fit_transform(sc['Participation in doubt sessions'])
sc.tail()
LR=LinearRegression()

# Clean percentage column
sc['Percentage of Attendance'] = sc['Percentage of Attendance'].str.replace('%', '').astype(float)

# Define input and output variables
ind = sc[['Enter_Your_Name', 'What is the dialy hours of study?', 'Usage of mobile phone while studying', 
          'Sleep Hours before exam?', 'Percentage of Attendance', 'Participation in doubt sessions']]
dep = sc['Final Score']

# Fit the model
LR.fit(ind, dep)

with open('/content/Final_scorepickle.pkl','wb') as f:
  pickle.dump(LR,f)
print("model trained and saved as 'Final_Score.pkl'")

from sklearn.metrics import mean_squared_error
val=LR.predict(ind)
# Use a regression metric like mean_squared_error
# You can choose other regression metrics like mean_absolute_error or r2_score
mse = mean_squared_error(dep, val)
print(f"Mean Squared Error: {mse}")

n=int(input("Enter_Your_Name"))
hr=int(input("What is the dialy hours of study?"))
mo=int(input("Usage of mobile phone while studying"))
sl=int(input("Sleep Hours before exam?"))
at=int(input("Percentage of Attendance"))
d=int(input("Participation in doubt sessions"))

ans=LR.predict([[n,hr,mo,sl,at,d]])
```

### Results

The model provides predictions for students' final scores, which can help educators identify students needing additional support.

### Requirements

- pandas
- scikit-learn

Feel free to use or modify this project as a starting point for your own predictive analytics in education!
