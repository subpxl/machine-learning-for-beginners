import pandas as pd

df = pd.DataFrame({
    "Color":["Red","Blue","Green"]
})

encoded = pd.get_dummies(df)

print(encoded)

#    Color_Blue  Color_Green  Color_Red
# 0       False        False       True
# 1        True        False      False
# 2       False         True      False