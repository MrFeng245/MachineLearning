import numpy as np 
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

class BackPropagation():
    def __init__(self, X_train, y_train, epochs, alpha) -> None:
        '''
        多分类问题：
            输入：784
            输出：10
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.alpha = alpha
        self.w, self.b = [], []
        self.m, self.n = X_train.shape
        self.layers = None
        self.loss_list = []
        self.y_hat = []

    def sigmoid(self, z):
        '''实现sigmoid函数'''
        return 1. / (1. + np.exp(-z))
    
    def derivation_sigmoid(self, z):
        '''实现sigmoid函数的导数'''
        return np.exp(-z) / np.square(1. + np.exp(-z))
    
    def tanh(self, z):
        '''实现双曲正切函数'''
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    def derivation_tanh(self, z):
        '''实现tanh的导数'''
        return 4. / np.square(np.exp(z) + np.exp(-z))

    def sequential(self, layers):
        '''顺序创建一个神经网络，计算f_x'''
        self.layers = layers
        for layer in layers:
            units = layer[0]
            self.w.append(np.random.randn(self.n, units) / np.sqrt(self.n))
            self.b.append(np.random.randn(units) / np.sqrt(self.n))
            self.n = units
    
    def dense(self, units, activation):
        '''传递消息'''
        return [units, activation]
    
    def fit(self):
        '''gradient descent'''
        for epoch in range(self.epochs):
            z, a = self.forword_prop()
            dj_dw, dj_db = self.back_prop(z, a)
            # 梯度下降
            for i in range(len(self.layers)):
                self.w[i] = self.w[i] - self.alpha * dj_dw[i]
                self.b[i] = self.b[i] - self.alpha * dj_db[i]

            loss = self.compute_loss(a[-1])
            self.loss_list.append(loss)
            if epoch % 100 == 99:
                print(f'---------第{epoch+1}次迭代--------------')
                print(f'loss={loss}')
        # 绘图
        self.plot_img('loss.png')

    def compute_loss(self, yhat):
        '''计算loss'''
        row = self.y_train.shape[0]
        return np.sum(np.square(self.y_train - yhat)) / (2 * row)
    
    def forword_prop(self):
        '''前向传播：得到由x得到yhat'''
        z, a = [], [self.X_train]
        for i in range(len(self.layers)):
            w, b = self.w[i], self.b[i]
            z.append(np.dot(a[i], w) + b)
            if self.layers[i][1] == 'sigmoid':
                a.append(self.sigmoid(z[i]))
            elif self.layers[i][1] == 'tanh':
                a.append(self.tanh(z[i]))
            else:
                print('没有这个激活函数')
        return z, a

    def back_prop(self, z, a):
        '''反向传播：由y-yhat修正w和b'''
        dj_dw = [0 for _ in range(len(self.layers))]
        dj_db = [0 for _ in range(len(self.layers))]
        error = a[-1] - self.y_train
        for i in range(len(self.layers) - 1, -1, -1):
            if self.layers[i][1] == 'sigmoid':
                J = error * self.derivation_sigmoid(z[i])
            elif self.layers[i][1] == 'tanh':
                J = error * self.derivation_tanh(z[i])
            error = np.dot(J, self.w[i].T)
            row = J.shape[0]
            dj_dw[i] = np.dot(a[i].T, J) / row
            dj_db[i] = np.sum(J, axis=0) / row
        return dj_dw, dj_db
    
    def predict(self, x_test, y_test):
        a = x_test
        for i in range(len(self.layers)):
            z = np.dot(a, self.w[i]) + self.b[i]
            if self.layers[i][1] == 'sigmoid':
                a = self.sigmoid(z)
            if self.layers[i][1] == 'tanh':
                a = self.tanh(z)  
        # 对独热编码进行解法 
        yhat = np.argmax(a, axis=1)
        y = np.argmax(y_test, axis=1)
        amount = y_test.shape[0]
        accuracy = np.sum(yhat == y) / amount
        print(f'拟合后，模型的精确度为{(accuracy * 100):.2f}%')

    def plot_img(self, name):
        '''绘制二维折线图'''
        x = [i for i in range(self.epochs)]
        y = self.loss_list
        plt.plot(x, y)
        plt.title(name, fontsize=24)
        plt.xlabel('iter', fontsize=14)
        plt.ylabel('cost', fontsize=14)
        plt.scatter(x[-1], y[-1], c='green', edgecolors='none', s=100)
        #plt.show()
        plt.savefig(f'BP/img/{name}', bbox_inches='tight')


def processed_data(train_img, train_label, test_img, test_label):
    # print(train_img.shape, test_img.shape)
    # print(train_img[0])
    # print(train_label[0])
    # plt.imshow(train_img[0])
    # plt.show
    train_img = train_img.reshape((60000, 28*28)).astype('float')
    test_img = test_img.reshape((10000, 28*28)).astype('float')
    train_label = to_categorical(train_label) # 转为独热编码
    test_label = to_categorical(test_label) # [0 1 0 0 0 0 0 0 0 0] => 1
    return train_img, train_label, test_img, test_label

if __name__ == '__main__':
    (train_img, train_label), (test_img, test_label) = mnist.load_data()
    # 数据归一化
    train_img = train_img / 255
    test_img = test_img / 255
    train_img, train_label, test_img, test_label = processed_data(train_img, train_label, test_img, test_label)
    # print(train_img.shape, train_label.shape, test_img.shape, test_label.shape)
    model = BackPropagation(train_img, train_label, epochs=1000, alpha=0.8)
    model.sequential([
        model.dense(15, 'tanh'),
        model.dense(10, 'sigmoid'),
    ])
    model.fit()
    model.predict(test_img, test_label)