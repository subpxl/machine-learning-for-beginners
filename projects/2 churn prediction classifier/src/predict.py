
def predict_probability(model,x):
    probabilities =model.predict_proba(x)
    return probabilities
