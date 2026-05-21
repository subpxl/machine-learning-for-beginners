from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


model =SVC()

params = {
    "C":[0.1,1,10],
    "kernel":["linear","rbf"]
}

grid = GridSearchCV(model,params)

print(grid)