import numpy as np
from sklearn.datasets import load_iris


class DecisionTree():
    def __init__(self, X_train, y_train, branch_name, max_depth) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.k = self.X_train.shape[1] # k类特征
        self.branch_name = branch_name
        self.max_depth = max_depth

    def split_dataset(self, data, featrue):
        '''以data的均指作为阈值划分数据集'''
        data = np.array(data)
        threshold = np.median(data[:, featrue])
        left_data, right_data = [], []    
        for i in range(data.shape[0]):
            if data[i][featrue] <= threshold:
                left_data.append(data[i])
            else:
                right_data.append(data[i])
        return left_data, right_data, threshold 

    def compute_entropy(self, data):
        '''计算熵'''
        entropy = 0.
        count = np.zeros(3) # 统计3类鸢尾花在数据集中对应的数量
        for i in range(len(data)):
            count[self.y_train[i]] += 1
        for i in range(3):
            term = count[i] / len(data)
            if term != 0:
                entropy += (term * np.log2(term))
        return -entropy
    
    def compute_information_gain(self, data, left_data, right_data):
        '''计算信息增益'''
        # feature: 0 1 2 3 中的某个
        left_entropy = self.compute_entropy(left_data)
        right_entropy = self.compute_entropy(right_data)
        w_left = len(left_data) / len(data)
        w_right = len(right_data) / len(data)
        weighted_entropy = w_left * left_entropy + w_right * right_entropy                                             
        information_gain = self.compute_entropy(data) - weighted_entropy
        return information_gain

    def choose_best_feature(self, data):
        '''返回一个最佳特征及划分划分阈值'''
        best_featrue, best_threshold, best_info_gain = -1, -1, -1
        for featrue in range(self.k):
            left_data, right_data, threshold = self.split_dataset(data, featrue)
            info_gain = self.compute_information_gain(data, left_data, right_data)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_featrue = featrue
                best_threshold = threshold
        return best_featrue, best_threshold

    def build_tree_recursive(self, data, branch_name, current_depth):
        '''构建决策树'''
        if current_depth == self.max_depth:
            formatting = " "*current_depth + "-"*current_depth
            result.append(data)
            print(f"{formatting} {branch_name} leaf node with indices")
            return
        best_feature, max_info_gain = self.choose_best_feature(data) 
        tree.append((current_depth, branch_name, best_feature))
        
        formatting = "-"*current_depth
        print(f"{formatting} Depth {current_depth}, {branch_name}: Split on feature: {best_feature}")
        
        left_data, right_data, threshold = self.split_dataset(data, best_feature)
        self.build_tree_recursive(left_data, "Left", current_depth+1)
        self.build_tree_recursive(right_data, "Right", current_depth+1)

    def predict(self, dataset):
        '''计算误差'''
        accuracy = 0.
        # 通过dataset列表找X_train的行索引
        for data in dataset:
            indices = []
            for value in data:
                index = np.where((self.X_train == value).all(1))[0]
                indices.append(index[0])
            count = np.bincount(self.y_train[indices])
            accuracy += max(count)
        return accuracy / len(self.y_train)

def load_data():
    iris = load_iris()
    iris_feature = iris.data
    iris_target = iris.target
    return iris_feature, iris_target

if __name__ == '__main__':
    tree, result = [], []
    X_train, y_train = load_data()
    model = DecisionTree(X_train, y_train, "Root", max_depth=5)
    model.build_tree_recursive(X_train, "Root", current_depth=0)
    accuracy = model.predict(result)
    print(f'模型预测精度为:{100 * accuracy:.2f}%')
