from sklearn.preprocessing import StandardScaler
import pandas as pd
data ={
    "Age":[20,25,30,35],
    "Salary":[20000,30000,40000,50000]
}

df = pd.DataFrame(data)

scaler = StandardScaler()

scaled = scaler.fit_transform(df)

print(scaled)

# visualize and compare this