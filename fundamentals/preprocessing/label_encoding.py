from sklearn.preprocessing import LabelEncoder
import pandas as pd


df = pd.DataFrame({
    "City":["Delhi","Mumbai","Delhi","Pune"]
})

encoder = LabelEncoder()

df["City_Encoded"]=encoder.fit_transform(df["City"])

print(df)
# todo what is fit transform