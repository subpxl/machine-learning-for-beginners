import pandas as pd

data = {
    "Age":[20,25,30],
    "Salary":[20000,30000,40000],
    "Experience":[1,3,5]
}

df = pd.DataFrame(data)

print(df.corr())

#             Age  Salary  Experience
# Age         1.0     1.0         1.0
# Salary      1.0     1.0         1.0
# Experience  1.0     1.0         1.0