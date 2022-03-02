import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_rows', 300)

train_df = pd.read_csv("Data/train_df_v4.csv")
test_df = pd.read_csv("Data/test_df_v4.csv")
ss_df = pd.read_csv("Data/sample_submission.csv")
submission_df = test_df['game_id']
train_df.drop(columns=['game_id', 'time_control_name', 'rating_rival'], inplace=True)
test_df.drop(columns=['game_id', 'time_control_name', 'rating_rival'], inplace=True)

#print(train_df.head())
print(test_df.info())
print(train_df.info())
print(ss_df.info())
train_df = sklearn.utils.shuffle(train_df)
#print(train_df.head())
X = train_df.drop(columns=['points']).values
X = preprocessing.scale(X)
y = train_df['points'].values

'''''
X_train = X[:-700]
y_train = y[:-700]

X_test = X[-700:]
y_test = y[-700:]


clf = LinearRegression()


clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

for X,y in zip(X_test, y_test):
    print(f"Model: {clf.predict([X])[0]}, Actual: {y}")
'''''

X_test = test_df.drop(columns=['points']).values
X_test = preprocessing.scale(X_test)
y_test = test_df['points'].values

clf = LinearRegression()
clf.fit(X, y)

points = []

for X, y in zip(X_test, y_test):
    points.append(clf.predict([X])[0])

print(len(points))

df = pd.DataFrame(points, columns=['points'])
submission_df = submission_df.to_frame()
submission_df['points'] = df['points']
print(submission_df.head())
submission_df.to_csv('Data/submission.csv', index=False)

