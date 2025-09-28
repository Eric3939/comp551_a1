import numpy as np


class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.weights = None 

    def fit_analytical(self, X, y):
        X = self._preprocess_X(X)
        self.weights = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def fit_gradient_descent(self, X, y, optimizer):
        X = self._preprocess_X(X)
        D = X.shape[1]
        w0 = np.zeros(D)
        self.weights = optimizer.optimize(self.compute_gradient, X, y, w0, self.compute_loss)
        return self

    def predict(self, X):
        X = self._preprocess_X(X)
        return X @ self.weights

    def compute_gradient(self, X, y, w):
        N = X.shape[0]
        yh = X @ w
        grad = .5 * (X.T @ (yh - y)) / N
        return grad
    
    def compute_loss(self, X, y, w):
        return 0.5 * np.sum((y - X @ w) ** 2)

    def _preprocess_X(self, X):
        # ensures 2d shape
        if X.ndim == 1:
            X = X[:, None]
        
        # add bias term
        if self.add_bias:
            N = X.shape[0]
            X = np.column_stack([X, np.ones(N)])
        return X



class LogisticRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.weights = None

    def fit(self, X, y, optimizer):
        X = self._preprocess_X(X)
        D = X.shape[1]
        w0 = np.zeros(D)
        self.weights = optimizer.optimize(self.compute_gradient, X, y, w0, self.compute_loss)
        return self

    def predict_proba(self, X):
        X = self._preprocess_X(X)
        return self._sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_gradient(self, X, y, w):
        N = X.shape[0]
        yh = self._sigmoid(X @ w)
        grad = (X.T @ (yh - y)) / N
        return grad

    def compute_loss(self, X, y, w):
        yh = self._sigmoid(X @ w)
        return np.sum(-y * np.log(yh) - (1 - y) * np.log(1 - yh))
    
    def _sigmoid(self, z):
        return 1. / (1 + np.exp(-z))

    def _preprocess_X(self, X):
        # ensures 2d shape
        if X.ndim == 1:
            X = X[:, None]
        
        # add bias term
        if self.add_bias:
            N = X.shape[0]
            X = np.column_stack([X, np.ones(N)])
        return X



class GradientDescent:
    def __init__(self, lr=.001, max_iters=1e4, epsilon=1e-8, record_history=False, record_loss=False):
        self.lr = lr
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.record_history = record_history
        self.record_loss = record_loss
        if record_history:
            self.w_history = []
        if record_loss:
            self.loss_history = []

    def optimize(self, gradient_fn, X, y, w0, loss_fn=None):
        w = w0
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(X, y, w)
            w = w - self.lr * grad
            if self.record_history:
                self.w_history.append(w)
            if self.record_loss and loss_fn is not None:
                current_loss = loss_fn(X, y, w)
                self.loss_history.append(current_loss)
            t += 1
        return w


class StochasticGradientDescent:
    def __init__(self, lr=.001, max_iters=1e4, batch_size=32, epsilon=1e-8, record_history=False, record_loss=False):
        self.lr = lr
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.record_history = record_history
        self.record_loss = record_loss
        if record_history:
            self.w_history = []
        if record_loss:
            self.loss_history = []

    def optimize(self, gradient_fn, X, y, w0, loss_fn=None):
        w = w0
        grad = np.inf
        t = 1
        N = X.shape[0]
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            # sample batch
            batch_indices = np.random.choice(N, self.batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # gradient descent
            grad = gradient_fn(X_batch, y_batch, w)
            w = w - self.lr * grad
            
            if self.record_history:
                self.w_history.append(w)
            if self.record_loss and loss_fn is not None:
                current_loss = loss_fn(X_batch, y_batch, w)
                self.loss_history.append(current_loss)
            t += 1
        return w
