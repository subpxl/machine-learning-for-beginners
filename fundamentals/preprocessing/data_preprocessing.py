import pandas as pd 
import numpy as np

data = {
    "Age":[22,25,np.nan,30],
    "Salary":[25000,30000,3200,np.nan],
    "Department":["HR","IT","IT","Finance"]
}

df = pd.DataFrame(data)

print("original data")
print(df)

# fill missing values 
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Salary"]=df["Salary"].fillna(df["Salary"].mean())


# encode categorical column
df = pd.get_dummies(df,columns=["Department"])

print("\nProcessed Data:")
print(df)