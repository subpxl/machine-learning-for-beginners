from sklearn.datasets import make_classification
from collections import Counter

x,y = make_classification(
    n_samples=1000,
    n_classes=2,
    weights=[0.95,0.05],
    random_state=42
)


print(Counter(y))

# balanuce using ovesamplaing

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(random_state=32)

x_resampled, y_resampled = oversample.fit_resample(x,y)

print(Counter(y_resampled))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_resampled,y_resampled)
