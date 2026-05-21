from skimage.feature import hog
from sklearn import svm
from sklearn.datasets import load_digits

digits = load_digits()

features = []

for image in digits.images:
    fd = hog(image,
             pixels_per_cell=(4, 4),
             cells_per_block=(2, 2))
    features.append(fd)

X = features
y = digits.target

clf = svm.SVC()

clf.fit(X, y)

print(clf.predict([X[0]]))