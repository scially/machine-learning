import numpy as np
from numpy import linalg
from scipy.special import expit
import struct

class NeuralWork(object):
    "三层神经网络,批量训练法"

    def __init__(self, input_nodes, output_nodes, learn_rate, hidden_nodes, epochs, shuffle=True, debug=False):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.learn_rate = learn_rate
        self.hidden_nodes = hidden_nodes
        self.shuffle = shuffle
        self.w1 = None
        self.w2 = None
        self.epochs = epochs
        self.costs = None
        self.debug = debug

    def init_wight(self):
        self.w1 = np.random.normal(loc=0., scale=0.1, size=(self.hidden_nodes, self.input_nodes + 1))
        self.w2 = np.random.normal(loc=0., scale=0.1, size=(self.output_nodes, self.hidden_nodes + 1))

    def add_bais_unit(self, x):
        """
        对X添加偏置
        :param x: np.ndarray
        :return:  np.ndarray
        """
        if x.ndim == 1:
            x_copy = np.r_[1, x]
            return x_copy
        else:
            raise AttributeError('ndim must be 1')

    def forward(self, x, w1, w2):
        """
        前向传播
        :param x: x: array, shape=[input_nodes]
        :return: tupe(array,array,array,array,array)
                 最后一层结果,均包含偏置
        """

        # step1: 计算隐藏层输出结果
        # [hidden_nodes,input_nodes+1] dot [input_nodes+1,]
        # -> [hidden_nodes,]
        a1 = self.add_bais_unit(x)
        z2 = np.dot(w1, a1)
        a2 = self.sigmod(z2)

        # step2: 计算输出层输出结果
        # [output_nodes, hidden_nodes+1] dot [hidden_nodes+1,]
        # -> [output_nodes,]
        a2 = self.add_bais_unit(a2)
        z3 = np.dot(w2, a2)
        a3 = self.sigmod(z3)
        return a1, z2, a2, z3, a3

    def predict(self, X):
        if X.ndim == 1:
            a1, z2, a2, z3, a3 = self.forward(X, self.w1, self.w2)
            return np.argmax(a3)
        else:
            Y = []
            for x in X:
                a1, z2, a2, z3, a3 = self.forward(x, self.w1, self.w2)
                Y.append(np.argmax(a3))
            return np.array(Y)

    def fit(self, X_train, Y_train, X_valid, Y_valid, print_progress=True):

        self.costs = []
        self.init_wight()

        for idx in range(self.epochs):
            if self.shuffle:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train = X_train[indices]
                Y_train = Y_train[indices]
                Y_train_onehot = self.onehots(Y_train)

            cost = 0
            for x, y in zip(X_train, Y_train_onehot):
                a1, z2, a2, z3, a3 = self.forward(x, self.w1, self.w2)
                cost += self.get_cost(a3, y)
                w1_update, w2_update = self.get_gradient(a1, z2, a2, z3, a3, y)

                # check gradient
                if self.debug:
                    gradient_sigma = self.gradient_checked(x, y, w1_update, w2_update)
                    if gradient_sigma <= 1e-7:
                        print('OK: {0}'.format(gradient_sigma))
                    elif gradient_sigma <= 1e-4:
                        print('Warnning: {0}'.format(gradient_sigma))
                    else:
                        print('Error: {0}'.format(gradient_sigma))
                self.w1 -= self.learn_rate * w1_update
                self.w2 -= self.learn_rate * w2_update

            Y_train_pred = self.predict(X_train)
            Y_valid_pred = self.predict(X_valid)
            train_acc = np.sum(Y_train_pred == Y_train) / len(Y_train_pred)
            valid_acc = np.sum(Y_valid_pred == Y_valid) / len(Y_valid_pred)
            if print_progress:
                print('Epoch: {0}/{1} | Costs:{2:.3f} | Train Acc:{3:.3f} | Valid Acc:{4:.3f}'.
                      format(idx + 1, self.epochs, cost / len(Y_train_onehot), train_acc, valid_acc),
                      end='\r')
            self.costs.append(cost / len(Y_train_onehot))

    def get_cost(self, a3, y):
        """
        代价函数
        :param a3: array, shape=[output_nodes,]
        :param y: array, shape[output_nodes,]
        :return: int
                 代价函数
        """
        cost = -y * np.log(a3) - (1. - y) * np.log(1. - a3)
        return cost.sum()

    def get_gradient(self, a1, z2, a2, z3, a3, y):
        """

        :param a1: array, shape=[input_nodes,]
        :param z2: array, shape=[hidden_nodes,]
        :param a2: array, shape=[hidden_nodes+1,]
        :param z3: array, shape=[output_nodes,]
        :param a3: array, shape=[output_nodes,]
        :param y: array, shape=[output_nodes,]
        :return: tuple(array,array)
                 包含偏置项的权重
        """
        # step1: 计算输出层和隐藏层的误差项，包含偏置项
        output_sigma = a3 - y
        # [hidden_nodes+1, output_nodes] dot [output_nodes,]
        # ->[hidden_nodes+1,]
        hidden_sigma = np.dot(self.w2.T, output_sigma)

        # step2：计算隐藏层和输入层代价函数梯度
        # [output_nodes,1] dot [1,hidden_nodes+1]
        # -> [output_nodes,hidden_nodes+1]
        hidden_sigma[1:] = hidden_sigma[1:] * self.sigmod_derivative(z2)
        hidden_grad = np.dot(output_sigma[:, np.newaxis], a2[np.newaxis, :])
        # [hidden_nodes,1] dot [1,input_nodes+1]
        # -> [hidden_nodes,input_nodes+1]
        input_grad = np.dot(hidden_sigma[1:, np.newaxis], a1[np.newaxis, :])

        return input_grad, hidden_grad

    def gradient_checked(self, x, y, input_grad, hidden_grad, ep = 1e-6):
        wight_grad1 = np.zeros(self.w1.shape)
        wight_grad2 = np.zeros(self.w2.shape)
        wight_eps = np.zeros(self.w1.shape)
        # 对w1每一项求偏导
        for i in range(self.w1.shape[0]):
            for j in range(self.w1.shape[1]):
                wight_eps[i,j] = ep
                a1, z2, a2, z3, a3 = self.forward(x, self.w1-wight_eps, self.w2)
                cost1 = self.get_cost(a3, y)
                a1, z2, a2, z3, a3 = self.forward(x, self.w1 + wight_eps, self.w2)
                cost2 = self.get_cost(a3, y)
                wight_grad1[i,j] = (cost2 - cost1) / (2 * ep)
                wight_eps[i,j] = 0.

        # 对w2每一项求偏导
        wight_eps = np.zeros(self.w2.shape)
        for i in range(self.w2.shape[0]):
            for j in range(self.w2.shape[1]):
                wight_eps[i,j] = ep
                a1, z2, a2, z3, a3 = self.forward(x, self.w1, self.w2-wight_eps)
                cost1 = self.get_cost(a3, y)
                a1, z2, a2, z3, a3 = self.forward(x, self.w1, self.w2+wight_eps)
                cost2 = self.get_cost(a3, y)
                wight_grad2[i,j] = (cost2 - cost1) / (2 * ep)
                wight_eps[i,j] = 0.
        # 计算误差
        wight_grad = np.hstack((wight_grad1.flatten(), wight_grad2.flatten()))
        grad = np.hstack((input_grad.flatten(), hidden_grad.flatten()))
        norm1 = linalg.norm(wight_grad - grad)
        norm2 = linalg.norm(wight_grad)
        norm3 = linalg.norm(grad)
        return norm1 / (norm2 + norm3)

    def sigmod(self, x):
        return expit(x)

    @staticmethod
    def onehots(y):
        onehot = np.zeros((len(y), 10))
        for i, x in enumerate(y):
            onehot[i][x] = 1
        return onehot

    def sigmod_derivative(self, x):
        """
        sigmod导函数
        :param x: array or float
        :return: array or float
        """
        return self.sigmod(x) * (1 - self.sigmod(x))

#############
### Train ###
#############
		
lables_train = None
images_train = None
lables_test = None
images_test = None
with open('train-labels.idx1-ubyte','rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    lables_train = np.fromfile(lbpath, dtype = np.uint8)

with open('train-images.idx3-ubyte', 'rb') as imgpath:
    magic,num,rows,cols=struct.unpack('>IIII',imgpath.read(16))
    images_train = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 784)

with open('t10k-labels.idx1-ubyte','rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    lables_test = np.fromfile(lbpath, dtype = np.uint8)

with open('t10k-images.idx3-ubyte', 'rb') as imgpath:
    magic,num,rows,cols=struct.unpack('>IIII', imgpath.read(16))
    images_test = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 784)


# nn = NeuralNetMLP(n_hidden=50,
#                   l2=0,
#                   epochs=20,
#                   eta=0.0005,
#                   minibatch_size=1,
#                   shuffle=True,
#                   seed=1)
# nn.fit(X_train=images_train,
#        y_train=lables_train,
#        X_valid=images_test,
#        y_valid=lables_test)

nw = NeuralWork(input_nodes=784,
                output_nodes=10,
                learn_rate=0.0001,
                hidden_nodes=50,
                shuffle=True,
                debug=True,
                epochs=100)
nw.fit(images_train[:5], lables_train[:5],
       images_train[55000:], lables_train[55000:])