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
        if X.shape[1] != len(w):
            X = self._preprocess_X(X)
        N = X.shape[0]
        yh = X @ w
        grad = .5 * (X.T @ (yh - y)) / N
        return grad
    
    def compute_loss(self, X, y, w):
        if X.shape[1] != len(w):
            X = self._preprocess_X(X)
        N = X.shape[0]
        return 0.5 * np.sum((y - X @ w) ** 2) / N

    def _preprocess_X(self, X):
        # ensures 2d shape
        if X.ndim == 1:
            X = X[:, None]
        
        # add bias term
        if self.add_bias:
            # Check if bias column already exists
            if not np.allclose(X[:, -1], 1.0):  # If last column is not all ones
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
        if X.shape[1] != len(w):
            X = self._preprocess_X(X)
        N = X.shape[0]
        yh = self._sigmoid(X @ w)
        grad = (X.T @ (yh - y)) / N
        return grad

    def compute_loss(self, X, y, w):
        if X.shape[1] != len(w):
            X = self._preprocess_X(X)
        N = X.shape[0]
        epsilon = 1e-15  # Small value to avoid log(0)
        yh = self._sigmoid(X @ w)
        yh = np.clip(yh, epsilon, 1 - epsilon)
        return np.sum(-y * np.log(yh) - (1 - y) * np.log(1 - yh)) / N
    
    def _sigmoid(self, z):
        return 1. / (1 + np.exp(-z))

    def _preprocess_X(self, X):
        # ensures 2d shape
        if X.ndim == 1:
            X = X[:, None]
        
        # add bias term
        if self.add_bias:
            # Check if bias column already exists
            if not np.allclose(X[:, -1], 1.0):  # If last column is not all ones
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


# class StochasticGradientDescent:
#     def __init__(self, lr=.001, max_iters=1e4, batch_size=32, epsilon=1e-8, record_history=False, record_loss=False):
#         self.lr = lr
#         self.max_iters = max_iters
#         self.batch_size = batch_size
#         self.epsilon = epsilon
#         self.record_history = record_history
#         self.record_loss = record_loss
#         if record_history:
#             self.w_history = []
#         if record_loss:
#             self.loss_history = []

#     def optimize(self, gradient_fn, X, y, w0, loss_fn=None):
#         w = w0
#         grad = np.inf
#         t = 1
#         N = X.shape[0]
#         while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
#             # sample batch
#             batch_indices = np.random.choice(N, self.batch_size, replace=False)
#             X_batch = X[batch_indices]
#             y_batch = y[batch_indices]
            
#             # gradient descent
#             grad = gradient_fn(X_batch, y_batch, w)
#             w = w - self.lr * grad
            
#             if self.record_history:
#                 self.w_history.append(w)
#             if self.record_loss and loss_fn is not None:
#                 current_loss = loss_fn(X_batch, y_batch, w)
#                 self.loss_history.append(current_loss)
#             t += 1
#         return w

class StochasticGradientDescent:
    def __init__(self, lr=.001, max_epochs=1000, batch_size=32, epsilon=1e-8, record_history=False, record_loss=False):
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.record_history = record_history
        self.record_loss = record_loss
        if record_history:
            self.w_history = []
        if record_loss:
            self.loss_history = []
        self.epoch_history = []

    def optimize(self, gradient_fn, X, y, w0, loss_fn=None):
        w = w0
        N = X.shape[0]
        n_batches = int(np.ceil(N / self.batch_size))
        
        epoch = 0
        
        while epoch < self.max_epochs:
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process all batches in this epoch
            for batch_idx in range(n_batches):
                # Get current batch
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradient and update weights
                grad = gradient_fn(X_batch, y_batch, w)
                w = w - self.lr * grad
                
                # Record weight history if requested (per iteration)
                if self.record_history:
                    self.w_history.append(w.copy())
            
            # Record loss at the end of each epoch (on full dataset)
            if self.record_loss and loss_fn is not None:
                current_loss = loss_fn(X, y, w)  # Use full dataset for epoch loss
                self.loss_history.append(current_loss)
                self.epoch_history.append(epoch)
            
            # Check convergence at the end of each epoch
            if np.linalg.norm(grad) < self.epsilon:
                break
                
            epoch += 1
        
        return w