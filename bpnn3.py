from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp, sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Database():
    def __init__(self, db_file):
        self.filename = db_file
        self.dataset = list()

    def load_csv(self):
        data = pd.read_csv(self.filename)
        data = np.array(data)
        self.dataset = data.tolist()

    def dataset_str_to_float(self):
        col_len = len(self.dataset[0])
        for row in self.dataset:
            for col in range(col_len):
                row[col] = float(row[col])

    def dataset_minmax(self):
        self.minmax = list()
        self.minmax = [[min(column), max(column)] for column in zip(*self.dataset)]

    def normalize_dataset(self): # 数据归一化
        self.dataset_minmax()
        for row in self.dataset:
            for i in range(len(row)):
                row[i] = (row[i] + 1 - self.minmax[i][0]) / (self.minmax[i][1] - self.minmax[i][0] + 1)

    def inverse_dataset(self, data):
        inverse_data = list()
        minvalue = self.minmax[-1][0]
        maxvalue = self.minmax[-1][1]
        for i in range(len(data)):
            d = data[i] * (maxvalue - minvalue + 1) + minvalue - 1
            inverse_data.append(int(d))
        return inverse_data

    def get_dataset(self):  # 构建数据集
        self.load_csv()
        self.dataset_str_to_float()
        self.normalize_dataset()
        return self.dataset


class BP_Network():
    # 初始化神经网络，n_inputs是输入层神经元数，n_hidden是隐含层神经元数，n_outputs是输出层神经元数, num_hidden_layer是隐含层数量
    def __init__(self, n_inputs, n_hidden, n_outputs, num_hidden_layer):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.num_hidden_layer = num_hidden_layer
        self.network = list()
        for i in range(num_hidden_layer):
            # 生成隐含层的各个神经元的权值，结果是一个元素为字典类型的列表
            if i == 0:
                hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
            else:
                hidden_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
            self.network.append(hidden_layer)
        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)

    # 计算神经元的激活值（加权和）
    def activate(self, weights, inputs):
        activation = weights[-1]  # 最后一个权值，即偏置
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    # 定义激活函数
    def transfer(self, activation):
        return 1.0/(1.0 + exp(-activation))

    # 定义神经网络的正向传播，为神经元计算输出 ,row是输入向量
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:  # 遍历每一层
            new_inputs = []
            for neuron in layer:  # 遍历层中的每个神经元
                activation = self.activate(neuron['weights'], inputs)  # 计算神经元加权和
                neuron['output'] = self.transfer(activation)  # 计算并存储神经元的输出
                new_inputs.append(neuron['output'])
            inputs = new_inputs  # 更新inputs向量 ，准备下一轮迭代
        return inputs  # 返回输出层的输出向量

    # 定义激活函数的导数
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # 反向传播误差信息, 计算纠偏并存储在神经元中
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):  # 从输出层反向计算纠偏
            layer = self.network[i]
            errors = list()  # 误差向量
            if i != len(self.network) - 1:  # 若为隐含层
                for j in range(len(layer)):  # 遍历当前层神经元
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['responsibility'] * neuron['weights'][j])
                    errors.append(error)
            else:  # 若为输出层
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['responsibility'] = errors[j] * self.transfer_derivative(neuron['output'])  # 计算并存储纠偏

    # 根据误差，更新网络权重
    def _update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:  # 若不是第一个隐含层，则更新 inputs为前一层的输出
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['responsibility'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['responsibility']

    # 根据指定的训练周期训练网络,train是训练数据集**************对于交通流预测，输出层只有一个神经元
    def train_network(self, train):
        for epoch in range(self.n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                #expected = [0 for i in range(self.n_outputs)]
                #expected[row[-1]] = 1
                expected = [row[-1]]   #对于交通流预测，输出层只有一个神经元，期望值只有一个
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self._update_weights(row, self.l_rate)
            print('>迭代周期=%d, 误差=%.3f' % (epoch, sum_error))

    # 预测
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs[0]

    # 利用随机梯度递减策略训练网络
    def back_propagation(self, train, test):
        self.train_network(train)
        predictions = list()
        for row in test:
            prediction = self.predict(row)
            predictions.append(prediction)
        return predictions

    # 将数据集划分为k等份用于交叉检验,k为参数n_folds
    def cross_validation_split(self, n_folds):
        dataset_split = list()
        dataset_copy = list(self.dataset)
        fold_size = int(len(self.dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    #计算平均相对误差MRE
    def mean_relative_error(self, yp, yt):
        rel_error = np.abs(np.array(yt) - np.array(yp)) / np.array(yt)
        mean_rel_error = np.average(rel_error)
        return mean_rel_error

    #计算均等系数
    def equality_coefficient(self, yp, yt):
        ypp = np.array(yp)
        ytt = np.array(yt)
        ypt = ytt - ypp
        ec = 1 - sqrt(np.dot(ypt, ypt)) / (sqrt(np.dot(ypp, ypp)) + sqrt(np.dot(ytt, ytt)))
        return ec

    #计算准确度
    def accuracy(self, yp, yt):
        return 1 - self.mean_relative_error(yp, yt)

    #用交叉分割的块来评估BP算法
    def cross_evaluate_algorithm(self, dataset, n_folds, l_rate, n_epoch):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.dataset = dataset
        folds = self.cross_validation_split(n_folds)
        for fold in folds:  #交叉检验
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.back_propagation(train_set, test_set)
            actual = [row[-1] for row in fold]
            mre = self.mean_relative_error(predicted, actual)
            acc = self.accuracy(predicted, actual)
            ec = self.equality_coefficient(predicted, actual)
            print('测试验证：MRE= %.3f ,Accuracy=%.3f, EC=%.3f' % (mre, acc, ec))

    # 将数据集划分为训练集和测试集
    def validation_split(self):
        dataset_split = list()
        train_set = list(self.dataset)
        test_size = int(len(self.dataset) * 0.2)  # 20%用于测试
        test_set = list()
        while len(test_set) < test_size:
            index = randrange(len(train_set))  # 随机抽取测试样本
            test_set.append(train_set.pop(index))
        dataset_split.append(train_set)
        dataset_split.append(test_set)
        return dataset_split

    #BP算法基本验证评估
    def evaluate_algorithm(self, dataset, l_rate, n_epoch):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.dataset = dataset
        folds = self.validation_split()
        train_set = folds[0]
        test_set = folds[1]
        predicted = self.back_propagation(train_set, test_set)
        actual = [row[-1] for row in test_set]
        mre = self.mean_relative_error(predicted, actual)
        acc = self.accuracy(predicted, actual)
        ec = self.equality_coefficient(predicted, actual)
        print('测试验证：MRE= %.3f ,Accuracy=%.3f, EC=%.3f' % (mre, acc, ec))
        return predicted, actual


seed(2)
filename = 'E:/dataset/data_mrows.csv'
DB = Database(filename)
dataset = DB.get_dataset()
n_inputs = len(dataset[0]) - 1
n_hidden = 8
n_outputs = 1
num_hidden_layser = 2
BP = BP_Network(n_inputs, n_hidden, n_outputs, num_hidden_layser)
l_rate = 0.4
n_folds = 5
n_epoch = 600
predicted, actual = BP.evaluate_algorithm(dataset, l_rate, n_epoch)
predicted = DB.inverse_dataset(predicted)
actual = DB.inverse_dataset(actual)
x = range(1, len(actual) + 1)
plt.plot(x, actual, color='r', label='actual output')
plt.plot(x, predicted, color='b', label='predicted output')
plt.title('BP Algorithm')
plt.xlabel('Samples', fontproperties='simhei', fontsize=10)
plt.ylabel('Outputs', fontproperties='simhei', fontsize=10)
plt.legend()
plt.show()





















