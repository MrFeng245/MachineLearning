import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


class CollaborativeFiltering(nn.Module):
    def __init__(self, watch, rating, test_data, features=10, epochs=100, alpha=0.1):
        super(CollaborativeFiltering, self).__init__()
        self.watch = watch
        self.rating = rating
        self.test_data = test_data
        self.epochs = epochs
        self.alpha = alpha # 学习率
        self.rows, self.columns = watch.shape
        # 初始化w,x,b
        self.w = nn.Parameter(torch.randn(self.rows, features), requires_grad=True)
        self.x = nn.Parameter(torch.randn(self.columns, features), requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.rows, 1), requires_grad=True)

    def forward(self):
        '''前向传播'''
        f = torch.matmul(self.w, self.x.T) + self.b
        return f
    
    def plot_loss_img(self, loss_list):
        '''画loss曲线'''
        x = [i for i in range(self.epochs)]
        plt.plot(x, loss_list)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss of Collaborative Filtering")
        plt.savefig('img/loss.png', dpi=120, bbox_inches='tight')
        # plt.show()

    def recommend(self, w, x, b):
        '''用训练后的w,x,b实现推荐功能'''
        length = len(self.test_data)
        true_positive_num = 0
        false_positive_num = 0
        for i in range(length):
            try:
                user_index = rating_table.columns.get_loc(self.test_data[i, 0]) 
                anime_index = rating_table.index.get_loc(self.test_data[i, 1])
            except:
                continue
            predict_score = np.sum(np.multiply(w[anime_index], x[user_index])) + b[anime_index, 0]
            print(f"第{i+1}个预测评分:{predict_score:.1f},实际评分:{test_data[i, 2]}")
            # true:预测正确，false:预测错误; positive:正类
            if predict_score >= 7 and test_data[i, 2] >= 7:
                true_positive_num += 1
            elif predict_score < 7 and test_data[i, 2] >= 7:
                false_positive_num += 1
        print(f"true_positive_num:{true_positive_num}")
        print(f"false_positive_num:{false_positive_num}")
        print(f'召回率:{(100 * true_positive_num / (true_positive_num + false_positive_num)):.2f}%')

    def train(self, model):
        '''训练w,x,b'''
        rating = torch.tensor(self.rating, requires_grad=True)
        watch = torch.tensor(self.watch, requires_grad=True)
        # 设置损失函数
        loss_func = F.mse_loss
        # 设置优化器
        opt = torch.optim.Adam(model.parameters(), lr=self.alpha)
        # 开始训练
        loss_list = []
        for epoch in range(self.epochs):
            loss = loss_func(model() * watch, rating) # 此处调用forward()
            loss.backward()
            opt.step() # 此处优化学习率
            opt.zero_grad()
            loss_list.append(loss.data)

        model.plot_loss_img(loss_list)
        para = model.state_dict() # 以字典的形式存放训练过程中的参数
        w = para['w'].numpy()
        x = para['x'].numpy()
        b = para['b'].numpy()
        model.recommend(w, x, b)

def load_data():
    '''导入训练数据'''
    table = pd.read_csv('CollaborativeFiltering/data/rating.csv', na_values=[-1])
    # DataFrame类型切片方式
    train_data = table.iloc[0:100000, :]
    test_data = table.iloc[100000:100500, :]
    # 取表中的值
    test_data = test_data.values
    # 重构train_data
    rating_table = train_data.pivot_table(index='anime_id', columns='user_id', values='rating')
    rating = rating_table.values
    rating[np.where(np.isnan(rating))] = 0
    # 通过是否为-1(nan)确定该用户是否观看过此动漫
    watch = copy.deepcopy(rating)
    watch[np.where(np.isnan(watch))] = 0
    watch[np.where(watch != 0)] = 1
    return rating_table, rating, watch, test_data

if __name__ == "__main__":
    rating_table, rating, watch, test_data = load_data()
    model = CollaborativeFiltering(watch, rating, test_data, features=8, epochs=80, alpha=0.1)
    model.train(model)
