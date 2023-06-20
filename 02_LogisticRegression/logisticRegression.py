import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self, X, y, theta, lambda_) -> None:
        '''
        参数：
            X shape(n, 3) 第三列均为0
            y shape(n, 1) 
            theta shape(3, 1) [[w1], [w2], [b]]
            lambda_ 标量
        '''
        self.X = X
        self.y = y
        self.theta = theta
        self.lambda_ = lambda_

    def sigmoid(self, z):
        '''实现sigmoid函数'''
        g = 1 / (1 + np.exp(-z))
        return g
    
    def compute_f(self):
        '''根据模型参数得到预测值'''
        z = np.matmul(self.X, self.theta)
        f_wb = self.sigmoid(z)
        return f_wb
    
    def cost_function(self):
        '''实现cost方法'''
        w = self.theta[:, 0:2]
        m = self.X.shape[0]
        f_wb = self.compute_f()
        regular = (self.lambda_ * np.sum(w * w)) / (2 * m) 
        total_cost = -self.y * np.log(f_wb) - (1 - self.y) * np.log(1 - f_wb) + regular
        J_wb = (np.sum(total_cost) / m)
        return J_wb
    
    def gradient_descent(self, alpha, iters): 
        '''实现梯度下降'''
        m = self.X.shape[0]
        cost = np.zeros(iters)
        tmp = np.zeros(self.theta.shape)
        parameters = len(self.theta)
        for i in range(iters):
            error = self.compute_f() - self.y
            for j in range(parameters):
                term = error * self.X[:, j].reshape(-1, 1)
                regular = (self.lambda_ * np.sum(self.theta[j, 0])) / m 
                tmp[j, 0] = self.theta[j, 0] - (alpha * np.sum(term) / m) + regular
                self.theta = tmp
            cost[i] = self.cost_function()
        return self.theta, cost
    
    def compute_error(self, theta):
        '''计算实际和预估间的误差'''
        error, m = 0., self.X.shape[0]
        yhat = np.matmul(self.X, theta)
        error = np.sum((yhat > 0.5) == self.y)
        return error / m
    
def plot_decision_img(name, X, y, theta):    
    '''绘制决策边界'''
    # 绘制点
    x_0 = X[np.where(y == 0)[0], :]
    x_1 = X[np.where(y == 1)[0], :]
    plt.scatter(x_0[:, 0], x_0[:, 1])
    plt.scatter(x_1[:, 0], x_1[:, 1])
    # 绘制曲线
    plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
    plot_y = -1. * (theta[0][0] * plot_x + theta[2][0]) / theta[1][0]
    print(plot_x, plot_y)
    plt.plot(plot_x, plot_y)
    # 设置图表标题，并给坐标轴加标签
    plt.title(name)
    plt.show()

def plot_loss_img(name, y):
    '''绘制loss折线图'''
    x=np.arange(0, len(y))
    plt.plot(x, y)
    plt.title(name, fontsize=24)
    plt.xlabel('iter', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.scatter(x[-1], y[-1], c='green', edgecolors='none', s=100)
    plt.show()
    
    
def test_one():
    data = np.loadtxt('data/ex2data1.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2:3]
    # 特征缩放
    X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    X = np.hstack((X, np.ones((len(y), 1))))
    theta = np.array([[0], [0], [0]])
    model = LogisticRegression(X=X, y=y, theta=theta, lambda_=0)
    theta, cost = model.gradient_descent(0.9, 1000)
    print(f'拟合后的参数为w=[{theta[0,0]}, {theta[1,0]}], b={theta[2,0]}')
    error = model.compute_error(theta)
    print(f'拟合后的模型精度为{error}')
    #plot_loss_img('loss of test1', cost)
    plot_decision_img("decision boundary of test2", X, y, theta)

def map_features(X):
    X_new = np.ones((X.shape[0],1))
    degree = 2  # 假设函数最高次方为2
    
    for i in range(1, degree + 1):  # 将x1,x2,x1^2,x1*x2,x2^2放入数组
        for j in range(0, i + 1):
            temp = X[:, 0] ** (i - j) * X[:, 1] ** j
            X_new = np.hstack((X_new, temp.reshape(-1, 1)))
    return X_new
    
def test_two():
    data = np.loadtxt('data/ex2data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2:3]
    X = map_features(X)
    X = np.hstack((X, np.ones((len(y), 1))))
    theta = np.ones((7,1)).reshape(-1,1)
    model = LogisticRegression(X=X, y=y, theta=theta, lambda_=0.01)
    theta, cost = model.gradient_descent(0.9, 1000)
    print(f'拟合后的参数为{theta}')
    error = model.compute_error(theta)
    print(f'拟合后的模型精度为{error}')
    plot_loss_img('loss of test2', cost)
    # plot_decision_img("decision boundary of test2", X[:, 0:2], y, theta)


if __name__ == '__main__':
    test_one()
    test_two()
