from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(model,x_test,y_test):

    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test
    )

    plt.title("confusion matrix")

    plt.show()

from sklearn.metrics import RocCurveDisplay

def plot_roc(model,x_test,y_test):
    RocCurveDisplay.from_estimator(
        model,
        x_test,
        y_test
    )

    plt.title("roc curve")

    plt.show()