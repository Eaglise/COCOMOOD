import numpy as np

class KNNRecommender:
    def __init__(self, metric="euclid"):
        self.metric = metric
        self.X = None
        self.distances = []

    def fit(self, X):
        self.X = X
        return self

    def eval(self, y, k):
        if self.metric == "manhattan":
            self.distances = np.sum(np.abs(self.X - y), axis=1)
        elif self.metric == "cosine":
            self.distances = np.array([self.__cosine_distance(x, y) for x in self.X])
        else:
            self.distances = np.linalg.norm(self.X - y, axis=1)
        idxs = self.__k_smallest_indices(self.distances, k)
        return [(idx, self.distances[idx]) for idx in idxs]

    @staticmethod
    def __k_smallest_indices(distances, k):
        return np.argsort(distances)[:k]

    @staticmethod
    def __cosine_distance(x, y):
        return np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
    
    @staticmethod
    def find_centroid_of_interest_from_scores(X, s):
        # s - вектор длины M, M - количество оценных тайтлов
        # X - матрица M x N, N - количество признаков
        return np.dot(s/np.linalg.norm(s), X)

    
# Пример
if __name__ == "__main__":
    X = np.array([
        [0, 0, 1],
        [2, 0, 0],
        [4, 4, 4],
        [1, 1, 0]
    ])
    y = np.array([1, 1, 0])
    res = KNNRecommender().fit(X).eval(y, 2)
    print(res)
    print(KNNRecommender.find_centroid_of_interest_from_scores(X, np.array([6,7,5,10])))