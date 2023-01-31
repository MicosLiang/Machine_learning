# coding: utf-8
# Author: L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special, scipy.spatial
from scipy.stats import norm

# 最优化函数求解器，对f(w1, w2,...)求取得最小/大值的参数w1,w2...
class Optimization:
    def __init__(self, cost, w, x : np.array = None, y : np.array = None, method: str = "gradient", st = None, aim : str = "min", iter : int = 100, eta : float = 1e-3, batch = 0) -> None:
        # 目标函数
        self.cost = cost
        # 可选的非优化参数（数据）
        self.x = x
        # 可选的非优化参数（标签）
        self.y = y
        # 优化参数
        self.w = w
        # 优化方法，梯度下降，遗传算法，粒子群
        self.method = method
        # 优化目标
        self.aim = aim
        # 梯度下降的训练批次
        self.batch = batch
        # 迭代次数
        self.iter = iter
        if self.batch:
            self.iter *= np.ceil(x.shape[0] / batch).astype("int")
        # 初始学习率
        self.eta = eta
        self.history = np.zeros(iter)
        pass

    def diff(self, w : np.array ,h = 1e-6):
        # 依次对wi求偏导
        def pdiff(f, x, y, w, i):
            w[i] += h
            z1 = f(x, y, w)
            w[i] -= 2*h
            z2 = f(x, y, w)
            w[i] += h
            return (z1 - z2) / (2*h)
        return np.array([pdiff(self.cost, self.x, self.y, w, i) for i in range(w.shape[0])])

    def gradient(self, beta1 = 0.9, beta2 = 0.999):
        # 初始化参数
        try:
            w = self.w()
        except Exception as e:
            w = self.w
        loss = 1e+10
        iter = self.iter
        eta = self.eta
        # Adam参数
        mt = np.zeros(w.shape)
        vt = np.zeros(w.shape)
        if self.batch:
            batch_size = np.floor(self.x.shape[0] / self.batch)
            batch_pos = 0
        self.history = np.zeros(iter)
        if self.aim == "min":
            aim = -1
        else:
            aim = 1
        while iter:
            # 逐批次训练，每次训练前打乱数据
            if self.batch:
                x = self.x[(batch_pos*self.batch):((batch_pos+1)*self.batch)]
                y = self.y[(batch_pos*self.batch):((batch_pos+1)*self.batch)]
                if batch_pos == batch_size:
                    indexs = np.arange(0,x.shape[0],1,dtype="int")
                    np.random.shuffle(indexs)
                    x = x[indexs]
                    y = y[indexs]
                    batch_pos = 0
                else:
                    batch_pos += 1
            else:
                x = self.x
                y = self.y
            # 计算目标函数当前解
            loss = self.cost(x, y, w)
            # 计算梯度
            gt = self.diff(w)
            # Adam寻找最适学习率
            mt = beta1 * mt + (1 - beta1) * gt
            vt = beta2 * vt + (1 - beta2) * pow(gt, 2)
            mt_hat = mt / (1 - pow(beta1, self.iter - iter + 1))
            vt_hat = vt / (1 - pow(beta2, self.iter - iter + 1))
            # 更新参数
            w += eta * mt_hat / (np.sqrt(vt_hat) + 1e-8) * aim
            # 记录迭代历史值
            self.history[self.iter - iter] = loss
            iter -= 1
        return w

    # 遗传算法
    def genetic(self, group_num = 1000, allow_var = 1, max_seed = 4):
        # 生成种群
        w = self.w()
        group = np.zeros([group_num, w.shape[0]])
        for i in range(group_num):
            group[i,:] = self.w()
        iter = self.iter
        self.history = np.zeros(iter)
        while iter:
            # 计算适合度
            losses = np.array([self.cost(self.x, self.y, group[i,:]) for i in range(group.shape[0])])
            # 筛选最优个体
            order = losses.argsort()
            if self.aim == "min":
                group = group[order[:group_num], :]
                losses = losses[order[:group_num]]
                #seeds = np.ceil((losses[group_num - 1] - losses) / ((losses[group_num-1] - losses[0])+1e-3) * max_seed + 0.01)
            else:
                group = group[order[-group_num:], :]
                losses = losses[order[-group_num:]]
                #seeds = np.ceil((losses - losses[0]) / ((losses[group_num-1] - losses[0])+1e-3) * max_seed + 0.01)
            # 每个个体产生配子
            indexs = np.random.choice([0,1], size=group.shape[1])
            x_seeds = group[:, indexs]
            np.random.shuffle(x_seeds)
            y_seeds = group[:, -indexs]
            np.random.shuffle(y_seeds)
            # 生成下一代种群
            sub_group = np.zeros(x_seeds.shape)
            sub_group[:, indexs] = x_seeds
            sub_group[:, -indexs] = y_seeds
            group = np.concatenate((group, sub_group), axis=0)
            # 记录迭代结果
            self.history[self.iter - iter] = losses[0]
            iter -= 1
        losses = np.array([self.cost(self.x, self.y, group[i,:]) for i in range(group.shape[0])])
        return group[losses.argsort()[0], :]

    # 粒子群算法
    def pso(self, group_num = 1000, wini = 0.9, wend = 0.4, c1 = 2, c2 = 2):
        # 初始化粒子群和方向
        w = self.w()
        group = np.zeros([group_num, w.shape[0]])
        vectors = np.zeros([group_num, w.shape[0]])
        # 每行是一个粒子
        for i in range(group_num):
            group[i,:] = self.w()
        for i in range(group_num):
            vectors[i,:] = self.w() * 1e-5
        # 参数初始化
        best_pos = group.copy()
        best_loss = np.array([self.cost(self.x, self.y, group[i,:]) for i in range(group_num)])
        vt = vectors.copy()
        iter = self.iter
        self.history = np.zeros(iter)
        while iter:
            # update group
            group = group + vt
            # update history
            losses = np.array([self.cost(self.x, self.y, group[i,:]) for i in range(group_num)])
            if self.aim == "min":
                indexs = (losses <= best_loss)
            else:
                indexs = (losses >= best_loss)
            best_pos[indexs,:] = group[indexs,:]
            best_loss[indexs] = losses[indexs]
            # 计算速度衰减系数
            wt = (iter / self.iter) * (wini - wend) + wend
            # 计算速度
            vt = wt * vectors
            vt += c1 * np.random.random(group_num).reshape([group_num,1]) * (best_pos - group)
            if self.aim == "min":
                vt += c2 * np.random.random(group_num).reshape([group_num,1]) * (group[np.argmin(losses),:] - group)
            else:
                vt += c2 * np.random.random(group_num).reshape([group_num,1]) * (group[np.argmax(losses),:] - group)
            c = c1 + c2
            vt *= 2 / np.abs(2 - c - np.sqrt(c**2 - 4*c))
            # history
            self.history[self.iter - iter] = np.min(losses)
            iter -= 1
        if self.aim == "min":
            return group[np.argmin(losses),:]
        return group[np.argmax(losses),:]
            
    def run(self):
        if self.method == "gradient":
            ans = self.gradient()
        elif self.method == "genetic":
            ans = self.genetic()
        elif self.method == "pso":
            ans = self.pso()
        plt.plot(range(self.iter), self.history)
        plt.show()
        return ans

# 全连接前馈神经网络
class Dense:
    class Activate:
        # activate，激活函数，f计算值，d计算偏倒数
        class Sigmoid:
            def f(self, x):
                return 1 / (1 - np.exp(-x))
            def d(self, x, backput):
                fx = self.f(x)
                return backput * (fx*(1-fx))
        class Relu:
            def f(self, x):
                ans = np.zeros(x.shape)
                indexs = x > 0
                ans[indexs] = x[indexs]
                return ans
            def d(self, x, backput):
                ans = np.zeros(x.shape)
                indexs = x > 0
                ans[indexs] = 1
                return backput * ans
        class Leaky_relu:
            def __init__(self, l = 1e-2) -> None:
                self.l = l
                pass
            def f(self, x):
                ans = np.zeros(x.shape)
                indexs = x > 0
                ans[indexs] = x[indexs]
                ans[~indexs] = x[~indexs] * self.l
                return ans
            def d(self, x, backput):
                ans = np.zeros(x.shape)
                indexs = x > 0
                ans[indexs] = 1
                ans[~indexs] = self.l
                return backput * ans
        class Tanh:
            def f(self, x):
                x = np.exp(-2*x)
                return (1 - x) / (1 + x)
            def d(self, x, backput):
                y = self.f(x)
                return backput * (1 - pow(y, 2))
        class Line:
            def f(self, x):
                return x
            def d(self, x, backput):
                return backput * np.ones(x.shape)
        class Softmax:
            def f(self, x):
                x = np.exp(x)
                x_sum = np.sum(x, axis=1)
                return x / x_sum.reshape([x_sum.shape[0], 1])
            def d(self, x, backput):
                y = self.f(x)
                return np.matmul(backput, np.diag(np.sum(y, axis=0)) - np.matmul(y.T,y))
    class Loss:
        # loss function，损失函数，f计算值，d计算偏倒数
        # 交叉熵
        class Cross_entropy:
            def f(self, x, y):
                return -np.sum(y*np.log(x))
            def d(self, x, y):
                return -y/x
        # 均方误差
        class Mse:
            def f(self, x, y):
                return 0.5 * np.sum(pow(y - x, 2)) / x.shape[0]
            def d(self, x, y):
                return (x - y) / x.shape[0]
    # 学习率优化器
    class Optimizer:
        class Adam:
            def __init__(self, shape, beta1 = 0.9, beta2 = 0.999, eta = 1e-3) -> None:
                self.mt = np.zeros(shape)
                self.vt = np.zeros(shape)
                self.eta = eta
                self.beta1 = beta1
                self.beta2 = beta2
                self.time = 0
                pass
            def cal(self, gt):
                self.time += 1
                self.mt = self.beta1 * self.mt + (1 - self.beta1) * gt
                self.vt = self.beta2 * self.vt + (1 - self.beta2) * pow(gt, 2)
                mt_hat = self.mt / (1 - pow(self.beta1, self.time))
                vt_hat = self.vt / (1 - pow(self.beta2, self.time))
                return self.eta * mt_hat / (np.sqrt(vt_hat) + 1e-8)
    # 神经网络隐藏层
    class Layer:
        def __init__(self,input, output, activate):
            self.shape = np.array([input, output], dtype="int")
            self.w = np.random.random(self.shape)
            # struct
            self.next = None
            self.front = None
            self.activate = activate
            # train
            self.x = None
            self.y = None
            self.optimizer = None
            pass

        def add(self, next_layer):
            self.next = next_layer
            return self.next

        # 前向传播
        def forward(self, input):
            self.x = input
            # 线性加权
            self.y = np.matmul(input, self.w)
            # 激活函数激活
            return self.next.forward(self.activate.f(self.y))
        
        # 反向传播
        def backward(self, backput):
            # 计算上级结果对激活函数导数
            deactivate = self.activate.d(self.y, backput)
            # 计算参数的偏导数
            diff = np.matmul(self.x.T,deactivate)
            # 将导数向上一层传播
            self.front.backward(np.matmul(deactivate, self.w.T))
            # 更新参数
            self.w -= self.optimizer.cal(diff)
    # 神经网络输入层
    class Input:
        def __init__(self) -> None:
            self.next = None
        def add(self, new_layer):
            self.next = new_layer
            return self.next
        def forward(self, input):
            return self.next.forward(input)
        def backward(self, backput):
            pass
    # 神经网络输出层
    class Output:
        def __init__(self, loss) -> None:
            self.front = None
            self.x = None
            self.loss = loss
            pass
        def forward(self, input):
            self.x = input
            return input
        def backward(self, y):
            print("loss: ", self.loss.f(self.x, y))
            self.front.backward(self.loss.d(self.x, y))
    # 数据预处理和后处理，归一化、正则化和反归一化、反正则化等操作
    class Data:
        def __init__(self, x, y, bias, method_x, method_y) -> None:
            self.x = x.copy()
            self.y = y.copy()
            # 是否偏置
            self.bias = bias
            # 数据预处理方法
            self.method_x = method_x
            self.method_y = method_y
            if len(self.y.shape) == 1:
                self.y = self.y.reshape([self.y.shape[0], 1])
            if len(self.x.shape) == 1:
                self.x = self.x.reshape([self.x.shape[0], 1])
            # record
            self.parms = {
                "min" : np.zeros(self.x.shape[1] + self.y.shape[1]),
                "max" : np.zeros(self.x.shape[1] + self.y.shape[1]),
                "mean" : np.zeros(self.x.shape[1] + self.y.shape[1]),
                "norm" : np.zeros(self.x.shape[1] + self.y.shape[1]),
                "std" : np.zeros(self.x.shape[1] + self.y.shape[1])
            }
            for k in range(self.x.shape[1]):
                self.parms["min"][k] = np.min(self.x[:,k])
                self.parms["max"][k] = np.max(self.x[:,k])
                self.parms["mean"][k] = np.mean(self.x[:,k])
                self.parms["norm"][k] = np.sqrt(np.sum(pow(x[:,k], 2)))
                self.parms["std"][k] = np.std(self.x[:,k])
                if method_x == "score":
                    self.x[:,k] = self.score(self.x[:,k], self.parms["mean"][k], self.parms["std"][k])
                elif method_x == "L2":
                    self.x[:,k] = self.L2(self.x[:,k], self.parms["norm"][k])
                elif method_x == "minmax":
                    self.x[:,k] = self.minmax(self.x[:,k], self.parms["min"][k], self.parms["max"][k])
                elif method_x == "line":
                    self.x[:,k] = self.line(self.x[:,k], self.parms["max"][k])
            for k in (np.arange(self.y.shape[1], dtype="int") + self.x.shape[1]):
                _k = k - self.x.shape[1]
                self.parms["min"][k] = np.min(self.y[:,_k])
                self.parms["max"][k] = np.max(self.y[:,_k])
                self.parms["mean"][k] = np.mean(self.y[:,_k])
                self.parms["norm"][k] = np.sqrt(np.sum(pow(self.y[:,_k], 2)))
                self.parms["std"][k] = np.std(self.y[:,_k])
                if method_y == "score":
                    self.y[:,_k] = self.score(self.y[:,_k], self.parms["mean"][k], self.parms["std"][k])
                elif method_y == "L2":
                    self.y[:,_k] = self.L2(self.y[:,_k], self.parms["norm"][k])
                elif method_y == "minmax":
                    self.y[:,_k] = self.minmax(self.y[:,_k], self.parms["min"][k], self.parms["max"][k])
                elif method_y == "line":
                    self.y[:,_k] = self.line(self.y[:,_k], self.parms["max"][k])
            if self.bias:
                self.x = np.c_[self.x, np.ones(self.x.shape[0])]
        # 数据还原操作
        def rechange_x(self, x):
            x = x.copy()
            if len(x.shape) == 1:
                x = x.reshape([x.shape[0], 1])
            for k in range(x.shape[1]):
                if self.method_x == "score":
                    x[:,k] = self.score(x[:,k], self.parms["mean"][k], self.parms["std"][k])
                elif self.method_x == "L2":
                    x[:,k] = self.L2(x[:,k], self.parms["norm"][k])
                elif self.method_x == "minmax":
                    x[:,k] = self.minmax(x[:,k], self.parms["min"][k], self.parms["max"][k])
                elif self.method_x == "line":
                    x[:,k] = self.line(x[:,k], self.parms["max"][k])
            if self.bias:
                x = np.c_[x, np.ones(x.shape[0])]
            return x
        def rechange_y(self, y):
            y = y.copy()
            if len(y.shape) == 1:
                y = y.reshape([y.shape[0], 1])
            for k in (np.arange(self.y.shape[1], dtype="int") + self.x.shape[1]):
                if self.bias:
                    k -= 1
                _k = k - self.x.shape[1]
                if self.method_y == "score":
                    y[:,_k] = self.score(self.y[:,_k], self.parms["mean"][k], self.parms["std"][k])
                elif self.method_y == "L2":
                    y[:,_k] = self.L2(self.y[:,_k], self.parms["norm"][k])
                elif self.method_y == "minmax":
                    y[:,_k] = self.minmax(self.y[:,_k], self.parms["min"][k], self.parms["max"][k])
                elif self.method_y == "line":
                    y[:,_k] = self.line(self.y[:,_k], self.parms["max"][k])
            return y
        def rescale_y(self, y):
            y = y.copy()
            if len(y.shape) == 1:
                y = y.reshape([y.shape[0], 1])
            for k in (np.arange(self.y.shape[1], dtype="int") + self.x.shape[1]):
                if self.bias:
                    k -= 1
                _k = k - self.x.shape[1]
                if self.method_y == "score":
                    y[:,_k] = self.descore(y[:,_k], self.parms["mean"][k], self.parms["std"][k])
                elif self.method_y == "L2":
                    y[:,_k] = self.deL2(y[:,_k], self.parms["norm"][k])
                elif self.method_y == "minmax":
                    y[:,_k] = self.deminmax(y[:,_k], self.parms["min"][k], self.parms["max"][k])
                elif self.method_y == "line":
                    y[:,_k] = self.deline(y[:,_k], self.parms["max"][k])
            return y
        # 中心化
        def score(self, x, _mean, _std):
            return (x - _mean) / _std
        # 正则化
        def L2(self, x, _norm):
            return x / _norm
        # 归一化
        def minmax(self, x, _min, _max):
            return (x - _min) / (_max - _min)
        # 线性归一化
        def line(self, x, _max):
            return x / _max
        def descore(self, x, _mean, _std):
            return (x * _std) + _mean
        def deL2(self, x, _norm):
            return x * _norm
        def deminmax(self, x, _min, _max):
            return x * (_max - _min) + _min
        def deline(self, x, _max):
            return x * _max

    # 神经网络对象init
    def __init__(self, x, y, loss = "mse", optimizer = "adam", bias = True, method_x = None, method_y = None) -> None:
        # 数据预处理
        self.data = self.Data(x, y, bias, method_x, method_y)
        self.x = self.data.x
        self.y = self.data.y
        # 推算神经网络结构
        self.now_shape = self.x.shape[1]
        self.start = self.Input()
        self.end = self.start
        # 保存超参数
        self.loss = loss
        self.optimizer = optimizer
        self.state = 1
        pass
    # 神经网络添加隐藏层
    def add(self, out_shape, activate = "relu"):
        if self.state:
            new_layer = self.Layer(self.now_shape, out_shape, activate)
            self.now_shape = out_shape
            self.end = self.end.add(new_layer)
        return self
    # 神经网络固定模型并初始化
    def init(self):
        # 确保输出层维度和目标结果一直
        if self.end.shape[1] != self.y.shape[1]:
            print("end node shape not match with y")
            return
        # 允许用户自定义损失函数
        if self.loss == "mse":
            self.loss = self.Loss.Mse()
        elif self.loss == "cross_entropy":
            self.loss = self.Loss.Cross_entropy()
        # 添加输出层，连接损失函数
        self.end = self.end.add(self.Output(self.loss))
        # 为每一层隐藏层添加激活函数，允许用户自定义损失函数
        obj = self.start.next
        obj.front = self.start
        while obj != self.end:
            obj.next.front = obj
            if obj.activate == "sigmoid":
                obj.activate = self.Activate.Sigmoid()
            elif obj.activate == "relu":
                obj.activate = self.Activate.Relu()
            elif obj.activate == "line":
                obj.activate = self.Activate.Line()
            elif obj.activate == "softmax":
                obj.activate = self.Activate.Softmax()
            elif obj.activate == "leaky_relu":
                obj.activate = self.Activate.Leaky_relu()
            elif obj.activate == "tanh":
                obj.activate = self.Activate.Tanh()
            # 为此层初始化学习率优化器
            if self.optimizer == "adam":
                obj.optimizer = self.Optimizer.Adam(obj.shape)
            obj = obj.next
        # 将神经网络切换到可训练状态
        self.state = 0
    # 训练，跑多少遍数据集和批次大小
    def train(self, epoch = 5000, batch = 32):
        # 训练前必须初始化
        if self.state:
            return False
        iter_time = np.floor(self.x.shape[0] / batch).astype("int")
        while epoch:
            # 打乱数据
            indexs = np.arange(0,self.x.shape[0],dtype="int")
            np.random.shuffle(indexs)
            self.x = self.x[indexs]
            self.y = self.y[indexs]
            # 迭代
            for iter in range(iter_time):
                # 前向传播更新梯度
                self.start.forward(self.x[(iter*batch):((iter+1)*batch)])
                # 反向传播更新参数
                self.end.backward(self.y[(iter*batch):((iter+1)*batch)])
            epoch -= 1
    # 输出预测结果
    def predict(self, x, y, acc = None):
        # 将预测数据进行和训练集一样的预处理
        x = self.data.rechange_x(x)
        y = self.data.rechange_y(y)
        # 推算结果
        ans = self.start.forward(x)
        loss = self.end.loss.f(ans, y)
        # 还原结果
        ans = self.data.rescale_y(ans)
        # 如果用户自定义了精确度计算函数，计算精确度
        if acc:
            acc = acc(ans, y)
        return ans, loss, acc

# 聚类
class Cluster():
    def __init__(self, x, category = 0, iter = 1000) -> None:
        self.x = x.copy()
        # 预期分类数
        self.category = category
        self.iter = iter
        pass

    # 计算_x每一行对_y每一行的距离矩阵
    def cal_dis(self, _x, _y):
        def dis(x):
            y = np.floor(x / _y.shape[0]).astype("int")
            x = x % _y.shape[0]
            return np.sqrt(np.sum((_x[y,:] - _y[x,:])**2))
        dis = np.frompyfunc(dis, 1, 1)
        dismaxtrix = np.arange(0,_x.shape[0]*_y.shape[0]).reshape([_x.shape[0],_y.shape[0]])
        dismaxtrix = dis(dismaxtrix)
        return dismaxtrix

    # 无监督聚类
    def k_means(self):
        if self.category <= 0:
            return
        # 选择初始聚类中心
        centers = np.zeros([self.category, self.x.shape[1]])
        centers[0,:] = self.x[0,:]
        # 聚类矩阵
        distance = self.cal_dis(self.x, self.x)
        # 选择未选择的，到目前已选择的所有点距离之和最大的点
        mask = np.ones(self.x.shape[0], dtype="bool")
        mask[0] = False
        indexs = np.arange(self.x.shape[0], dtype="int")
        for k in range(1,self.category):
            index = np.argmax(np.sum(distance[mask, :][:,~mask], axis=1))
            index = indexs[mask == 1][index]
            mask[index] = False
            centers[k,:] = self.x[index,:]
        # 迭代更新聚类中心
        iter = self.iter
        while True:
            # 计算各个点到蕨类中心距离
            distance = self.cal_dis(centers, self.x)
            # 分别归类到最近的类别中
            category = np.argmin(distance, axis=0)
            # 更新聚类中心
            new_centers = np.zeros([self.category, self.x.shape[1]])
            for each in range(self.category):
                new_centers[each,:] = np.mean(self.x[category == each,:], axis=0)
            # 聚类中心不再更新时停止
            if np.sum(np.abs(new_centers - centers)) / np.sum(np.abs(centers)) > 1e-4:
                centers = new_centers
            else:
                break
        return centers, category

# 评价数据降维
class Evalute():
    class Entropy:
        def __init__(self, x, types) -> None:
            weight = np.zeros([x.shape[1],1])
            # 1 离散， 0连续
            for t in range(x.shape[1]):
                if types[t]:
                    weight[t,1] = self.entropy_weight(x[:,t])
                else:
                    weight[t,1] = self.entropy_weight_knn(x[:,t])
            weight /= np.log(x.shape[0])
            weight = 1 - weight
            weight /= np.sum(weight)
            self.weight = weight
            self.value = np.matmul(x, weight)
        
        # 估计连续变量的熵
        def entropy_weight_knn(self, x, k = 3):
            if len(x.shape) == 1:
                x = x.reshape([x.shape[0], 1])
            kdtree = scipy.spatial.KDTree(x)
            dis, _ = kdtree.query(x, k + 1, eps=0, p=2)
            dis = dis[:, -1]
            dis[dis < 0] = 0
            if np.all(dis == 0):
                return 0
            dis = np.sum(np.log(2 * dis)) / x.shape[0]
            entropy = -scipy.special.digamma(k) + scipy.special.digamma(x.shape[0]) + (0.5 * np.log(np.pi) - np.log(np.sqrt(np.pi))) + dis
            return entropy

        # 离散变量的熵
        def entropy_weight(self, x):
            p = np.bincount(x) / x.shape[0]
            p = p[p > 0]
            return -np.sum(p*np.log(p))

        # 没写好
        def entropy_bins(self, x):
            raise
            x = np.sort(x)
            def bin_p(_x):
                return _x.shape[0] / (x.shape[0] * (np.max(_x) - np.min(_x)))
            def iter(_x, bp):
                Nk = _x.shape[0]
                _l = _x[:int(Nk/2)]
                _lp = bin_p(_l)
                _r = _x[int(Nk/2):]
                _rp = bin_p(_r)
                if np.abs(_lp - bp) > 1e-2 and _l.shape[0] > 3:
                    _la = iter(_l, _lp)
                else:
                    _la = _lp
                if np.abs(_rp - bp) > 1e-2 and _r.shape[0] > 3:
                    _ra = iter(_r, _rp)
                else:
                    _ra = _rp
                return np.log(_la) + np.log(_ra)
            return -iter(x, bin_p(x)) / x.shape[0]

    # 模糊评价
    class Fuzzy:
        def __init__(self, x, level_func, level_num) -> None:
            member_ship = np.zeros([x.shape[0], x.shape[1], level_num])
            for i in range(x.shape[1]):
                # 隶属度矩阵
                member_ship[:,i,:] = level_func(x[:,i])
            pass

    # 主成分分析
    class PCA:
        def __init__(self, matrix : np.array, dim = 2) -> None:
            coef = np.corrcoef(matrix.T)
            # vector每一列是val对应的特征向量
            val, vector = np.linalg.eig(coef)
            # 线性变换
            self.vector = vector
            matrix = np.matmul(matrix, vector)
            self.matrix = matrix
            self._val = val
        
        def val(self, dim = None, minmax = False):
            if not dim:
                dim = self.val.shape[0]
            I = np.matmul(self.matrix[:, :dim], self.val[:dim].T).T
            if minmax:
                I = (I - np.min(I)) / (np.max(I) - np.min(I))
            return I

        def change(self, x):
            return np.matmul(x, self.vector)

# 图
class Graph:
    # 图的一个节点
    class Node:
        def __init__(self, id, g,connect = [], data = None) -> None:
            # 链接的点
            self.connect = connect
            # 该点数据
            self.data = data
            # 该点ID
            self.id = id
            self.g = g
            pass

    # 初始化图，toself为一个节点自己到自己的转移概率，default默认填充值
    def __init__(self, _to_self = np.inf, data = None, default = np.inf) -> None:
        # 邻接矩阵
        self.weight = np.array([[_to_self]],dtype="float")
        self.mid = 1
        # 维护数据结构
        self.nodes = [self.Node(0, self, connect = [] if _to_self == -1 else [0], data = data)]
        self.default = default
        pass

    # 添加一个节点，_go为从该点出发链接的节点，form为其它点出发连接到这个点的点
    def add(self, _go = [], _from = [], _to_self = np.inf, data = None):
        # 初始化新的邻接行
        go_weight = (np.ones(self.weight.shape[1]) * self.default)
        from_weight = (np.ones(self.weight.shape[0] + 1) * self.default).reshape([self.weight.shape[0]+1, 1])
        for i in _go:
            go_weight[i[0]] = i[1]
        go_weight = go_weight.reshape([1, self.weight.shape[1]])
        self.weight = np.r_[self.weight, go_weight]
        for i in _from:
            from_weight[i[0]] = i[1]
        from_weight[-1] = _to_self
        # 更新邻接矩阵
        self.weight = np.c_[self.weight, from_weight]
        # 维护数据结构
        node = self.Node(self.mid, self, connect = _go, data = data)
        self.mid += 1
        self.nodes.append(node)
        return self.mid - 1

    # 动态规划求图的最短路
    def dijkstra(self, start = 0):
        s = [start]
        v = [i for i in range(self.mid)]
        # start点到其他所有点的最短距离
        dist = self.weight[start,:].copy()
        dist[start] = 0
        while self.mid - len(s):
            # 找到一个最近的点
            left = np.setdiff1d(v, s)
            j = left[np.argmin(dist[left])]
            s.append(j)
            # 相当于从该点出发，再寻找最近的点
            for i in left:
                if dist[j] + self.weight[j,i] <= dist[i]:
                    dist[i] = dist[j] + self.weight[j,i]
        return dist

    # 求图的最大流
    def FF(self, start, end):
        max_flow = 0
        v = [i for i in range(self.mid)]
        weight = self.weight.copy()
        # 清除反向边
        for i in range(weight.shape[0]):
            for k in range(weight.shape[1]):
                if weight[i,k]:
                    weight[k,i] = 0
        # 找到一条增广路
        def find_route(i, _route = [], flow = [], get = 0):
            _route.append(i)
            if get:
                flow.append(get)
            if i != end:
                left = np.setdiff1d(v, _route)
                if len(left):
                    for k in left:
                        if weight[i,k] > 0:
                            the_route, the_flow = find_route(k, _route, flow, weight[i,k])
                            if the_route:
                                return the_route, the_flow
            else:
                return _route, flow
            return [], 0
        # 更新残存网络
        def update_weight(_route, _flow, i = 0):
            if _route[i] != end:
                weight[_route[i],_route[i+1]] -= _flow
                # 更新反向边
                weight[_route[i+1],_route[i]] += _flow
                update_weight(_route, _flow, i+1)
        route, flow = find_route(start)
        while route:
            min_flow = np.min(flow)
            max_flow += min_flow
            update_weight(route, min_flow)
            route, flow = find_route(start, [], [], 0)
        return max_flow

# 支持向量机
class SVM():
    def __init__(self, x, y, c=1, k = "line", iter = 300, g = 1):
        # 核操作，线性核和高斯核
        if k == "line":
            self.k = np.dot
        elif k == "gauss":
            self.g = g
            self.k = self.guass_k
        self.x = x.copy()
        self.y = y.copy()
        # 打乱数据
        indexs = np.arange(0,x.shape[0],1,dtype="int")
        np.random.shuffle(indexs)
        self.x = self.x[indexs,:]
        self.y = self.y[indexs]
        if len(self.y.shape) == 1:
            self.y = self.y.reshape(self.y.shape[0], 1)
        # 初始化参数
        self.c = c
        self.b = 0
        self.alpha = np.zeros([y.shape[0],1], dtype="float")
        self.iter = iter

    # 高斯核
    def guass_k(self, _x, _y):
        if len(_y.shape) == 1:
            _y = _y.reshape([_y.shape[0], 1])
        if len(_x.shape) == 1:
            by_axis = 0
            res = np.zeros([1,_y.shape[1]])
        else:
            by_axis = 1
            res = np.zeros([_x.shape[0],_y.shape[1]])
        for k in range(_y.shape[1]):
            _y_ = _y[:,k]
            res[:,k] = np.exp(-np.sqrt(np.sum((_x - _y_)**2, axis=by_axis)) / 2 * self.g**2)
        if len(_x.shape) == 1:
            return res[0,0]
        return res

    # 分段函数
    def sign(self, x):
        x[x > 0] = 1
        x[x < 0] = -1
        return x

    # 预测分类标签
    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape([x.shape[0], 1])
        else:
            x = x.T
        return self.sign(np.sum(self.k(self.x, x) * self.y * self.alpha, axis=0) + self.b)

    # 序列最小优化法训练SVM
    def smo(self):
        iter = self.iter
        while iter:
            for i in range(self.y.shape[0]):
                for k in range(self.y.shape[0]):
                    if i != k:
                        # 相当于求解一个二次规划问题
                        kii = self.k(self.x[i,:],self.x[i,:])
                        kkk = self.k(self.x[k,:],self.x[k,:])
                        kik = self.k(self.x[i,:],self.x[k,:])
                        ei = -self.y[i] + self.predict(self.x[i,:])
                        ek = -self.y[k] + self.predict(self.x[k,:])
                        k_old = self.alpha[k]
                        i_old = self.alpha[i]
                        eta = kii + kkk - 2 * kik
                        if eta > 0:
                            self.alpha[k] += self.y[k] * (ei - ek) / eta
                        else:
                            print("warning, eta <= 0")
                            continue
                        if self.y[i] == self.y[k]:
                            L = max(0, k_old + i_old - self.c)
                            H = min(self.c, k_old + i_old)
                        else:
                            L = max(0, k_old - i_old)
                            H = min(self.c, self.c + k_old - i_old)
                        if self.alpha[k] > H:
                            self.alpha[k] = H
                        elif self.alpha[k] < L:
                            self.alpha[k] = L
                        self.alpha[i] += self.y[i]*self.y[k]*(self.alpha[k] - k_old)
                        bi = -ei - (self.alpha[i] - i_old)*self.y[i]*kii - (self.alpha[k] - k_old)*self.y[k]*kik + self.b
                        bk = -ek - (self.alpha[k] - k_old)*self.y[k]*kkk - (self.alpha[i] - i_old)*self.y[i]*kik + self.b
                        if 0 < self.y[i] < self.c:
                            self.b = bi
                        elif 0 < self.y[k] < self.c:
                            self.b = bk
                        else:
                            self.b = (bi +  bk) / 2
            iter -= 1
        # 筛选出支持向量
        indexs = (self.alpha > 0)
        indexs = indexs[:,0]
        self.alpha = self.alpha[indexs]
        self.x = self.x[indexs, :]
        self.y = self.y[indexs]

# 决策树
class Tree():
    class Node:
        def __init__(self, mask1, mask2, char = -1, data = None, father = None) -> None:
            # 标记哪些数据对该节点不可见
            self.mask1 = mask1
            self.mask2 = mask2
            # 上级节点
            self.father = father
            # 该节点用于分类的数据和值
            self.char = char
            self.data = data
            # 该节点的子节点
            self.sons = []
            pass
        # 添加子节点
        def add(self, mask1, mask2, char, data):
            new_node = Tree.Node(mask1, mask2, char, data, self)
            self.sons.append(new_node)
            return new_node

    def __init__(self, x, y, type) -> None:
        self.x = x
        self.y = y
        self.type = type
        self.rows = np.arange(0,self.x.shape[0],1,dtype="int")
        self.cols = np.arange(0,self.x.shape[1],1,dtype="int")
        # 初始化根节点
        self.root = self.Node(np.ones(x.shape[0],dtype="bool"),np.ones(x.shape[1],dtype="bool"))
        pass

    # 计算离散变量熵
    def entropy(self, x):
        p = np.bincount(x) / x.shape[0]
        p = p[p > 0]
        return -np.sum(p*np.log(p))

# ID3决策树，继承自Tree对象
class ID3(Tree):
    def train(self):
        # 递归生成树
        self.root = self.iter(self.root)

    def iter(self, root):
        # 仍有可用数据
        if np.any(root.mask1) and np.any(root.mask2):
            # 已经完美分类则退出
            if np.all(self.y[mask1] == self.y[mask1][0]):
                return root
            # 选取条件熵最小的特征
            info_gain = [self.info_gain(root.mask1, k) for k in self.cols[root.mask2]]
            char = self.cols[root.mask2][np.argmin(info_gain)]
            # 生成子节点，递归
            for each in set(self.x[root.mask1,:][:,char]):
                mask1 = root.mask1.copy()
                mask2 = root.mask2.copy()
                mask1[self.x[:,char] != each] = False
                mask2[char] = False
                next_node = root.add(mask1, mask2, char, each)
                self.iter(next_node)
        return root

    # 计算条件熵
    def info_gain(self, i, k):
        w = []
        e = []
        for char in set(self.x[i,:][:,k]):
            indexs = self.x[i,:][:,k] == char
            w.append(np.sum(indexs))
            e.append(self.entropy(self.y[i][indexs]))
        w = np.array(w) / np.sum(w)
        return np.sum(w * np.array(e))

    # 预测
    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape([1,x.shape[0]])
        types = np.zeros(x.shape[0])
        for k in range(x.shape[0]):
            root = self.root
            _x = x[k,:]
            while len(root.sons):
                for son in root.sons:
                    if _x[son.char] == son.data:
                        root = son
                        break
            types[k] = np.argmax(np.bincount(self.y[root.mask1]))
        return types
            
# 决策树测试用例
# ID3处理离散值
# melon =np.array([
# ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
# # 2
# ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
# # 3
# ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
# # 4
# ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
# # 5
# ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
# # 6
# ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
# # 7
# ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
# # 8
# ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
# # 9
# ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
# # 10
# ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
# # 11
# ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
# # 12
# ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
# # 13
# ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
# # 14
# ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
# # 15
# ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
# # 16
# ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
# # 17
# ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
# ])
# melon_x = melon[:,:-1]
# melon_y = np.zeros(17, dtype="int")
# melon_y[:8] = 1
# tree = ID3(melon_x, melon_y, None)
# tree.train()
# print(np.sum(tree.predict(melon_x) == melon_y))

# c4.5,CART处理连续和离散变量
melon2 = np.array([
# 1
['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
# 2
['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
# 3
['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
# 4
['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
# 5
['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
# 6
['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
# 7
['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
# 8
['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
# 9
['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
# 10
['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
# 11
['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
# 12
['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
# 13
['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
# 14
['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
# 15
['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
# 16
['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
# 17
['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
])
melon2_x = melon2[:,:-1]
melon2_y = np.zeros(17, dtype="int")
melon2_y[:8] = 1

# SVM测试用例
# iris = pd.read_csv("iris.csv")
# iris_x = iris.values[:,1:-1].astype("float64")
# iris_y = np.zeros([150, 1])
# iris_y[iris.values[:,-1] == "versicolor"] = 1
# iris_y[iris.values[:,-1] == "virginica"] = -1
# iris_y = iris_y[50:]
# iris_x = iris_x[50:, :]
# svm = SVM(iris_x, iris_y, iter=10, k = "gauss")
# svm.smo()
# np.sum(svm.predict(iris_x) == iris_y[:,0])

# 图测试用例
# 最短路
# graph = Graph()
# graph.add(_from = [[0,2]])
# graph.add(_from = [[0,5],[1,1]])
# graph.add(_from = [[1,3],[2,1]])
# dist = graph.dijkstra(0)
# print(dist)
# 最大流
# graph = Graph(_to_self = 0, default= 0)
# graph.add(_from = [[0,8]], _to_self = 0)
# graph.add(_go = [[1,2]], _from = [[0,12]], _to_self = 0)
# graph.add(_from = [[1,6],[2,10]], _to_self = 0)
# graph.add(_go = [[3,2]], _from = [[1,10]], _to_self = 0)
# graph.add(_from = [[3,8],[4,10]], _to_self = 0)
# max_flow = graph.FF(0, 5)
# print(max_flow)

# 最优化测试用例
# def cost(x, y, w):
#     indexs = np.abs(w) > 500
#     if np.any(indexs):
#         return np.sum((np.abs(w[indexs]) - 500)*100)
#     return np.sum(-w * np.sin(np.sqrt(np.abs(w))))
# def genw():
#     return np.random.randint(-500, 500, 30).astype("float")
# op = Optimization(cost, genw, iter=800, method="pso")
# ans = op.run()
# print(ans)

# 分类测试用例
# iris = pd.read_csv("iris.csv")
# iris_x = iris.values[:,1:-1].astype("float64")
# iris_y = np.zeros([150, 3])
# iris_y[iris.values[:,-1] == "setosa", 0] = 1
# iris_y[iris.values[:,-1] == "versicolor", 1] = 1
# iris_y[iris.values[:,-1] == "virginica", 2] = 1

# 聚类
# data = iris_x
# cluster = Cluster(data, 4)
# centers, category = cluster.k_means()
# _data = Evalute.PCA(data)
# _centers = _data.change(centers)
# for k in set(category):
#     plt.scatter(_data.matrix[category==k,0],_data.matrix[category==k,1])
#     plt.scatter(_centers[k,0], _centers[k,1],marker='v')
# plt.show()

# model = Dense(iris_x, iris_y, loss="cross_entropy")
# model.add(8)
# model.add(3, activate="softmax")
# model.init()
# model.train(epoch=2000)
# print(model.predict(iris_x, iris_y, acc = lambda _x, _y : np.sum(np.argmax(_x, axis=1) == np.argmax(_y,axis=1))))

# 回归测试用例
# bh = pd.read_csv("house.csv")
# bh_x = bh.values[:,:-1].astype("float64")
# bh_y = bh.values[:,-1].astype("float64")
# bh_x = bh_x[~np.isnan(bh_y)]
# bh_y = bh_y[~np.isnan(bh_y)]
# model = Dense(bh_x, bh_y, method_x="score")
# model.add(128, "sigmoid")
# model.add(1, "line")
# model.init()
# model.train(epoch=10000)
# ans,loss,acc = model.predict(bh_x, bh_y)
# plt.plot(range(ans.shape[0]), ans)
# plt.plot(range(bh_y.shape[0]), bh_y)
# plt.legend(["predict", "true"])
# plt.show()


