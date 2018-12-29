import numpy as np
from scipy.special import expit


class LogisticReg:
    """
    逻辑斯蒂回归模型
    """
    def __init__(self, learn_rate, iter_nums, debug=False):
        """
        :param learn_rate: float, 学习速率
        :param iter_nums: int, 迭代次数
        """
        self.learn_rate = learn_rate
        self.iter_nums = iter_nums
        self.wight = None
        self.debug = debug
        self.costs = []

    def fit(self, X, Y):
        """
        :param X: array, shape=[simples, features]
        :param Y: array, shape=[featrues,]
        :return: self
        """

        simples, features = X.shape
        self.__init_wight(features + 1)

        # 经过测试，对于numpy的一维向量，numpy会自动进行广播（brodacast）扩展维度，不需要考虑维度
        # Y = Y.reshape((simples, 1))
        # 或者
        # Y = np.mat(Y)

        for i in range(self.iter_nums):
            cost = 0
            for x, y in zip(X, Y):
                cost += self.__partial_update_wight(x, y)

            # average cost
            if self.debug:
                print('{0}/{1} | cost:{2}'.format(i+1, self.iter_nums, cost/len(Y)))
            self.costs.append(cost / len(Y))
        return self

    def __init_wight(self, rows):
        self.wight = np.random.normal(loc=0., scale=.1, size=rows)

    def __partial_update_wight(self, x, y):
        """
        单个样本权重更新
        :param x: array, shape=[features,]
                特征值
        :param y: float
                实际值
        :return: float
        """
        output = self.net_input(x)
        _error = y - output
        # [features, ] dot [features, ]
        # -> [1,]
        self.wight[1:] += self.learn_rate * np.dot(x, _error)
        self.wight[0] += self.learn_rate * _error
        # cost
        return -y * np.log(output) - (1 - y) * np.log(1 - output)

    def net_input(self, x):
        """
        :param x: array, shape=[simples, features]
        :return: array, shape=[simples,]
        """
        # [simples, features] dot [features, 1]
        # -> [simples, 1]
        return expit(self.wight[0] + np.dot(x, self.wight[1:, ]))

    def predict(self, x):
        """
        :param x: array, shape=[simples, features]
        :return: array, shape=[simples,]
        """
        return np.where(self.net_input(x) >= 0.5, 1, 0)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('ex2data1.txt', delimiter=",", header=0)

    Y = df.iloc[0:100, 2].values
    X = df.iloc[0:100, [0, 1]].values

    # 对数据进行标准化
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    learn_rate = 0.1
    iter_nums = 20000
    gd = LogisticReg(learn_rate, iter_nums, debug=True)
    gd.fit(X, Y)