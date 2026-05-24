# Feature selection means:
# choosing the most useful input features for the model.
# Not every column helps prediction.
# Some features:
# add noise
# increase overfitting
# slow training
# reduce generalization

import pandas as pd

df = pd.DataFrame({
    "age":[25,30,35,40,45],
    "salary":[50000,60000,70000,80000,90000],
    "random_noise":[99,12,45,3,77],
    "target":[0,0,1,1,1]
})

correlation = df.corr()
print(correlation)
target_corr = correlation["target"]
print(target_corr)

selected_featues = ["age","salary"]
x = df[selected_featues]
print(x)


# method 2

print("method 2")
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif



class FeatureSelector:

    def __init__(self,dataframe,target):
        self.df  = dataframe
        self.targt =target

    def select_best_features(self,k=2):
        X = self.df.drop(columns=[self.targt])
        y = self.df[self.targt]

        selector = SelectKBest(
            score_func=f_classif,
            k=k
        )

        x_selected = selector.fit_transform(x,y)

        selected_columns = x.columns[selector.get_support()]

        return pd.DataFrame(
            x_selected,
            columns=selected_columns
        )
    

df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "salary": [50000, 60000, 70000, 80000, 90000],
    "random_noise": [99, 12, 45, 3, 77],
    "target": [0, 0, 1, 1, 1]
})


selector = FeatureSelector(df,"target")

result =selector.select_best_features(k=2)
print(result)