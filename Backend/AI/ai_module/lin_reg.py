import numpy as np

class LinearRegressor:
    def __init__(self, solver="analytic", batch_size=1, learning_rate=0.01, n_epochs=100, err=0.1):
        self.X = None
        self.y = None
        self.w = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.err = err
        self.solver = solver

    def fit(self, X, y):
        n_samples, _ = X.shape
        self.X = np.c_[X, np.ones(n_samples)]
        self.y = y
        if self.solver == "sgd":
            self.w = self.sgd(self.X, self.y, self.batch_size, self.learning_rate, self.n_epochs, self.err)
        else:
            self.w = self.analytic_solver(self.X, self.y)
        return self
    
    def predict(self, x):
        print(self.w)
        return np.append(x, 1) @ self.w
    
    @staticmethod
    def sgd(X, y, batch_size, learning_rate, n_epochs, err):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        cur_epoch = 1
        while cur_epoch < n_epochs:
            for i in range(batch_size, n_samples + 1, batch_size):
                # print(f"Epoch: {cur_epoch}", end="\t")
                X_batch = X[i - batch_size: i]
                y_batch = y[i - batch_size: i]
                e = np.dot(X_batch, w.T) - y_batch
                grad = 2 * np.dot(e, X_batch) / batch_size
                # print(f"Error: {np.linalg.norm(e)}")
                w -= learning_rate * grad
                cur_epoch += 1
        return w
                
    @staticmethod
    def analytic_solver(X, y):
        X_T = X.T
        return np.linalg.inv(X_T @ X) @ X_T @ y
    

if __name__ == "__main__":
    X = np.array([
        [0, 0, 1],
        [2, 0, 0],
        [4, 4, 4],
        [1, 1, 0]
    ])
    y = np.array([0, 1.5, 5, 1])
    res = LinearRegressor(solver="sgd").fit(X, y).predict(np.array([1,1,1]))
    print(res)