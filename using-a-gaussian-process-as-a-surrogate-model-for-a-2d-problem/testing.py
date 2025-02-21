import numpy as np
import scipy.linalg

class GaussianProcess:
    def __init__(self, kernel, noise=1e-6):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # Compute the kernel matrix
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        
        # Precompute inverse of K for later predictions
        self.K_inv = np.linalg.inv(K)
    
    def predict(self, X_test):
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + self.noise * np.eye(len(X_test))
        
        # Mean of the posterior
        mu_s = K_s.T @ self.K_inv @ self.y_train
        
        # Covariance of the posterior
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s
        
        return mu_s, cov_s

def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """Squared Exponential (RBF) Kernel for 2D inputs."""
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Sample Data
X_train = np.array([[-2, -2], [0, 0], [2, 2]])
y_train = np.array([[-1], [0], [1]])
X_test = np.array([[x, y] for x in np.linspace(-3, 3, 10) for y in np.linspace(-3, 3, 10)])

gp = GaussianProcess(kernel=rbf_kernel)
gp.fit(X_train, y_train)
mu_s, cov_s = gp.predict(X_test)

# Plot the results
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='r', label="Training Data")
ax.plot_trisurf(X_test[:, 0], X_test[:, 1], mu_s.ravel(), cmap='viridis', alpha=0.6)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Mean Prediction")
ax.legend()
plt.show()
