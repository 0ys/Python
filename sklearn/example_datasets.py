import pandas as pd
from sklearn import datasets
from sklearn import svm # Support vector machine

# iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)
print(digits.target)

# 모델 정의
clf = svm.SVC(gamma=0.001, C=100.)
# 모델 학습
clf.fit(digits.data[:-10], digits.target[:-10])
# 모델로 예측
pred = clf.predict(digits.data[-10:])

submission = pd.DataFrame({"prediction": pred})
submission.to_csv('sklearn/digits.csv')