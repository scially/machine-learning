import pandas as pd
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
df = pd.read_csv('ex2data1.txt', delimiter=",", header=0)

Y = df.iloc[0:100, 2].values
X = df.iloc[0:100, [0, 1]].values
lr.fit(X, Y)



# 对数据进行标准化
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
