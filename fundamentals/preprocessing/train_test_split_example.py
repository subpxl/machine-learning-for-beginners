from sklearn.model_selection import train_test_split
import numpy as np

x = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([1,0,1,0,1])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)

print("traininf data")
print(x_train)

print("testing data")
print(x_test)