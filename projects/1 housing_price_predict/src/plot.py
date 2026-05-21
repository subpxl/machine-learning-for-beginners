import numpy as np
import matplotlib.pyplot as plt



def plot_results(y_test,predictions):

    plt.scatter(y_test,predictions)

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")

    plt.show()

    
def plot_errors(y_test,predictions):
    errors = y_test-predictions
    plt.hist(errors,bins=20)

    plt.title("error distribution")
    plt.xlabel("error")
    plt.ylabel("frequency")

    plt.show()