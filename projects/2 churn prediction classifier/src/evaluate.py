from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(model, x_test, y_test):

    prediction = model.predict(x_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction)

    recall = recall_score(y_test, prediction)

    f1 = f1_score(y_test, prediction)

    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("f1 score", f1)