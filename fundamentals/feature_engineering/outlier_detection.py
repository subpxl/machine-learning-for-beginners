# Outliers are abnormal or extreme values.
# Outliers can:
# distort averages
# damage model training
# create unstable predictions
# Method 1 — IQR (Interquartile Range)

import pandas as pd

df = pd.DataFrame({
    "salary": [50000, 52000, 51000, 53000, 55000, 200000]
})

print(df)

q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)

iqr = q3-q1

lower_bound = q1-1.5*iqr
upper_bound = q3+1.5*iqr

outliers = df[
    (df["salary"] < lower_bound) |
    (df["salary"] > upper_bound)
]
print("outliers= ",outliers)


# complete example 

class OutlierDetector:
    def __init__(self,dataframe):
        
        self.df = dataframe

    def detect_outliers_iqr(self,column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)

        IQR =Q3-Q1

        lower = Q1-1.5*IQR
        upper = Q3+1.5*IQR

        outliers = self.df[
            (self.df[column] < lower) |
            (self.df[column] > upper)
        ]

        return outliers
    
    def remove_outliers_iqr(self,column):

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        self.df = self.df[
            (self.df[column] >= lower) &
            (self.df[column] <= upper)
        ]

        return self.df

df = pd.DataFrame({
    "salary": [50000, 52000, 51000, 53000, 55000, 200000]
})

detector = OutlierDetector(df)

print("Outliers:")
print(detector.detect_outliers_iqr("salary"))

print("\nCleaned Data:")
print(detector.remove_outliers_iqr("salary"))