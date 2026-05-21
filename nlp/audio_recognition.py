import librosa
from sklearn.ensemble import RandomForestClassifier

# load audio
audio, sr = librosa.load("sample.wav")

# extract MFCC features
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# aggregate
feature_vector = mfcc.mean(axis=1)

X_train = [feature_vector]
y_train = ["speech"]

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

print(clf.predict([feature_vector]))