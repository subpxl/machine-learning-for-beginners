from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

x,y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=10,
    random_state=43
)

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.2,
    random_state=32
)


model = LinearRegression()
model.fit(x_train,y_train)

y_pred= model.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mse**2
r2 = r2_score(y_test,y_pred)

print("mae",mae)
print("mse",mse)
print("rmse",rmse)
print("r2",r2)