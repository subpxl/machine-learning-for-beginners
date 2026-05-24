import pandas as pd

data = {
    "timestamp":[
        "2026-01-01 08:30:00",
        "2026-01-02 14:45:00",
        "2026-01-03 19:15:00",
        "2026-01-04 23:10:00"
    ]
}

df= pd.DataFrame(data)

df["timestamp"]=pd.to_datetime(df["timestamp"])

print(df)
df["year"]=df["timestamp"].dt.year
df["month"]=df['timestamp'].dt.month
df["day"]=df["timestamp"].dt.day

df["hour"]=df["timestamp"].dt.hour
df["minute"]=df["timestamp"].dt.minute

df["day_of_week"]=df["timestamp"].dt.day_of_week

df["is_weekend"]=df["day_of_week"].isin([5,6]).astype(int)

df["quarter"]=df["timestamp"].dt.quarter

print(df)