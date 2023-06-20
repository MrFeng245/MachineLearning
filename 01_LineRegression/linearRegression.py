import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, X, y, theta) -> None:
        '''
        Args:
            X shape(n,2/3)
            y shape(n) 每列内容：[价格]
            theta shape(2,1) (w, b)
        '''
        self.X = X
        self.y = y
        self.theta = theta

    def compute_model_output(self): 
        '''根据参数theta计算y的预测值'''
        yhat = np.matmul(self.X, self.theta)
        return yhat   
    
    def compute_cost(self):
        '''计算cost值'''
        error = np.matmul(self.X, self.theta) - self.y
        inner = np.power(error, 2)
        total_cost = np.sum(inner)/(2 * len(self.y))
        return total_cost
    
    def gradient_descent(self, alpha, iters):
        '''计算梯度下降，得到每次迭代后的cost及参数theta'''
        tmp = np.zeros(self.theta.shape)
        parameters = len(self.theta)
        cost = np.zeros(iters)
        # 迭代
        for i in range(iters):
            error = np.matmul(self.X, self.theta) - self.y
            for j in range(parameters):
                term = np.multiply(error, self.X[:, j]) # 矩阵内积
                tmp[j, 0] = self.theta[j, 0] - ((alpha / len(self.X) * np.sum(term)))
                self.theta = tmp
            self.theta = tmp
            cost[i] = self.compute_cost()
            
        return self.theta, cost
    
def plot_img(name, x, y):
    '''绘制二维折线图'''
    plt.plot(x, y)
    plt.title(name, fontsize=24)
    plt.xlabel('iter', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.scatter(x[-1], y[-1], c='green', edgecolors='none', s=100)
    plt.show()


def test_one_variable():
    data = np.loadtxt('data/data1.txt', delimiter=',')
    X = data[:, 0:1]
    y = data[:, 1:2]
    # 处理数据
    X = X / np.max(X)
    y = y / np.max(y)
    X = np.hstack((X, np.ones((len(y), 1))))
    w, b = 0, 0
    theta = np.array([[w], [b]])
    linear_regression = LinearRegression(X=X, y=y, theta=theta)
    theta, costs = linear_regression.gradient_descent(alpha=0.03, iters=50)
    # np.arange() 构造与costs等长的数组
    print(f'最终的参数为w={theta[0][0]}, b={theta[1][0]}')
    plot_img(name='one variable of cost', x=np.arange(0, len(costs)), y=costs)

def test_multiple_variable():
    data = np.loadtxt('data/data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2:3]
    # 处理数据
    X = X / np.max(X)
    y = y / np.max(y)
    X = np.hstack((X, np.ones((len(y), 1))))
    theta = np.array([[0], [0], [0]])
    linear_regression = LinearRegression(X=X, y=y, theta=theta)
    theta, costs = linear_regression.gradient_descent(0.01, 100)
    print(f'最终的参数为w=({theta[0][0]},{theta[1][0]}), b={theta[2][0]}')

    plot_img(name='multiple variable of cost', x=np.arange(0, len(costs)), y=costs)
    
if __name__ == '__main__':
    test_one_variable()
    test_multiple_variable()