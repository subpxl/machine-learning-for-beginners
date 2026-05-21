from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    )

import matplotlib.pyplot as plt

# Actual values
y_true = [
    0,0,0,0,0,
    1,1,1,1,1,
    0,1,0,1,1,
    0,0,1,1,0
]

# Predicted values
y_pred = [
    0,0,0,1,0,
    1,1,0,1,1,
    0,1,0,1,0,
    0,0,1,1,1
]

# Probability scores for class 1
y_prob = [
    0.1,0.2,0.3,0.7,0.9,
    0.8,0.4,0.9,0.2,0.85,
    0.1,0.95,0.2,0.8,0.45,
    0.1,0.3,0.9,0.88,0.7
]

# metrics 
accuracy = accuracy_score(y_true,y_pred)
precision =precision_score(y_true,y_pred)
recall = recall_score(y_true,y_pred)
f1=f1_score(y_true,y_pred)
auc = roc_auc_score(y_true,y_prob)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("AUC Score:", auc)


print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true,y_pred)
print("confision matrix")
print(cm)

display =ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(6,6))

display.plot(ax=ax)

plt.title("Confusion Matrix")

plt.show()


# roc curve
fpr,tpr ,thresholds = roc_curve(y_true,y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr)

plt.plot([0,1],[0,1],linestyle="--")

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")

plt.title("Roc curve")
plt.show()