import numpy as np
from sklearn.datasets import load_iris

class KMeans():
    def __init__(self, X_train, y_train, K, iters) -> None:
        self.K = K # 质心个数
        self.X = X_train
        self.y = y_train
        self.iters = iters

    def generate_centroids(self):
        '''根据特征个数生成质心array'''
        randidx = np.random.permutation(self.X.shape[0])
        centroids = self.X[randidx[:self.K]]
        return centroids

    def find_closest_centroids(self, centroids):
        '''求最接近质心的索引'''
        idx = np.zeros(self.X.shape[0], dtype=int)
        for i in range(len(idx)):
            distance = []
            for j in range(self.K):
                norm_ij = np.linalg.norm(self.X[i] - centroids[j]) # 求范数（即距离）
                distance.append(norm_ij)

            idx[i] = np.argmin(distance)
        return idx

    def compute_centroids(self, idx):
        '''计算k个质心的坐标'''
        
        centroids = np.zeros((self.K, self.X.shape[1]))
        for k in range(self.K):
            points = self.X[idx == k]
            centroids[k] = np.mean(points, axis=0)
        print(centroids)
        return centroids
    
    def run_kMeans(self):
        '''执行K-means算法'''
        centroids = self.generate_centroids() # 初始化质心数组
        idx = np.zeros(self.X.shape[0])
        for i in range(self.iters):
            print(f"K-Means iteration {i+1}/{self.iters}")
            idx = self.find_closest_centroids(centroids)
            centroids = self.compute_centroids(idx)

        return centroids, idx
    
    def accuracy(self, idx):
        '''精度'''
        accuracy = 0.
        right = np.where(self.y == idx)[0]
        print(right, len(right))
        accuracy = len(right) / len(self.y)
        print(f'本次运行时，模型精度为:{100 * accuracy:.2f}%')
        return accuracy

def load_data():
    iris = load_iris()
    iris_feature = iris.data
    iris_target = iris.target
    return iris_feature, iris_target

if __name__ == '__main__':
    tree, result = [], []
    X_train, y_train = load_data()
    model = KMeans(X_train, y_train, K=3, iters=25)
    max_accuracy = 0.
    n = 15
    best_centroids = []
    for i in range(n):
        centroids, idx = model.run_kMeans()
        accuracy = model.accuracy(idx)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_centroids = centroids
    print(f'经过{n}次随机质心，得到最佳精度为:{100 * max_accuracy:.2f}%\n\n最好的质心数组为:\n{best_centroids}')
