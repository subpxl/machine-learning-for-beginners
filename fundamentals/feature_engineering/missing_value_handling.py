import pandas as pd
import numpy as np

class MissingValueHandler:

    def __init__(self,dataframe):
        self.df = dataframe

    def show_missing(self):
        print(self.df.isnull().sum())
    
    def fill_mean(self,column):
        self.df[column]=self.df[column].fillna(
            self.df[column].mean()
        )

    
    def fill_median(self,column):
        self.df[column]=self.df[column].fillna(
            self.df[column].median()
        )

    def get_dataframe(self):
        return self.df
    
df = pd.DataFrame({
    "age":[25,20,np.nan,40],
    "salary":[50000,np.nan,70000,80000]
})
print("df with nan",df)

handler = MissingValueHandler(df)

handler.show_missing()

handler.fill_mean("age")
handler.fill_median("salary")

print(handler.get_dataframe())