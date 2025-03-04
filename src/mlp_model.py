import numpy as np
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class FFSNNetwork:
    def __init__(self, n_inputs, hidden_sizes=[10]):
        # Initialize the inputs
        self.nx = n_inputs
        self.ny = 1  # Output layer size (binary classification)
        self.nh = len(hidden_sizes)  # Number of hidden layers
        self.sizes = [self.nx] + hidden_sizes + [self.ny]
        
        # Initialize weights and biases
        self.W = {}
        self.B = {}
        for i in range(self.nh + 1):
            self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
            self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh + 1):
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1]) + self.B[i + 1]
            self.H[i + 1] = self.sigmoid(self.A[i + 1])
        return self.H[self.nh + 1]

    def grad_sigmoid(self, x):
        return x * (1 - x)
    
    def grad(self, x, y):
        self.forward_pass(x)
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        L = self.nh + 1
        self.dA[L] = (self.H[L] - y)
        for k in range(L, 0, -1):
            self.dW[k] = np.matmul(self.H[k - 1].T, self.dA[k])
            self.dB[k] = self.dA[k]
            self.dH[k - 1] = np.matmul(self.dA[k], self.W[k].T)
            self.dA[k - 1] = np.multiply(self.dH[k - 1], self.grad_sigmoid(self.H[k - 1]))

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):
        if initialise:
            for i in range(self.nh + 1):
                self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))
        
        if display_loss:
            loss = {}
        
        for e in range(epochs):
            dW = {}
            dB = {}
            for i in range(self.nh + 1):
                dW[i + 1] = np.zeros((self.sizes[i], self.sizes[i + 1]))
                dB[i + 1] = np.zeros((1, self.sizes[i + 1]))
            
            for x, y in zip(X, Y):
                self.grad(x, y)
                for i in range(self.nh + 1):
                    dW[i + 1] += self.dW[i + 1]
                    dB[i + 1] += self.dB[i + 1]
            
            m = X.shape[0]
            for i in range(self.nh + 1):
                self.W[i + 1] -= learning_rate * dW[i + 1] / m
                self.B[i + 1] -= learning_rate * dB[i + 1] / m
            
            if display_loss:
                Y_pred = self.predict(X)
                loss[e] = mean_squared_error(Y, Y_pred)
        
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()
