from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import numpy as np



def evaluate_model(model,x_test,y_test):
    
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test,predictions)
    mse= mean_squared_error(y_test,predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,predictions)
    

    print("MAE",mae)
    print("MSE",mse)
    print("RMSE",rmse)
    print("R2 score",r2)
