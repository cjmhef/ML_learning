# coding=utf-8
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv(open("logistic_regression1.csv"))
data = data.dropna(axis=0)
data = data.values
print(data.shape)
# X = data[:-50, 6:-1]
# y = data[:-50, -1]
# y = y.astype('int')
# X_temp = data[-100:, 6:-1]
# y_temp = data[-100:, -1]
X = data[:-10, [6, 8, 9, 11]]
y = data[:-10, -1]
y = y.astype('int')
X_temp = data[-10:, [6, 8, 9, 11]]
y_temp = data[-10:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
clf = LogisticRegression(solver="lbfgs").fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.predict(X_test))
print(y_test)
print("截距为：%s"%clf.intercept_)
print("参数为：%s"%clf.coef_)
print("不停电和停电的概率分别为：%s"%clf.predict_proba(X_test[0:5]))#[0不停电,1停电]
# 重过载时长/统计时长,平均三相不平衡度,重三相不平衡度/统计时长
#召回率、准确率、F值、支持度
