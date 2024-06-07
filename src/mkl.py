from kernel import Kernel
from svm import SVC
from svm import SMOSolver
import numpy as np
from tqdm import tqdm

class MKL:
    def __init__(self, kernels, C=0.5, max_iters=10):
        """MKL class to perform multiple kernel learning with MKLGL algorithm
        Args:
            kernels: list of Kernel objects, kernels to be used
            C: float, regularization parameter
            max_iters: int, number of iterations
        """
        self.kernels = kernels
        for kernel in self.kernels:
            assert isinstance(kernel, Kernel), "All kernels should be of type Kernel"

        self.C = C
        self.max_iters = max_iters
        self.K = []

    def getKs(self, X):
        for kernel in self.kernels:
            k = kernel(X, X)
            k = k / np.trace(k)
            self.K.append(k)

    def get_kernel(self, X, y):
        self.getKs(X)
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        # initialize the coefficients with mean
        lambdas = np.ones((int(num_classes*(num_classes - 1)/2), len(self.kernels))) / len(self.kernels)
        for p in range(self.max_iters):
            print("Iteration: ", p)
            n=0
            for i in range(num_classes):
                for j in range(i+1, num_classes):

                    idx = np.where((y == self.classes[i]) | (y == self.classes[j]))[0]
                    X_c= X[idx]
                    y_c = np.where(y[idx] == self.classes[i], -1, 1)
                    # compute the kernel matrix from last iteration
                    K = np.zeros((len(idx) ,len(idx)))
                    for k in range(len(self.kernels)):
                        K += lambdas[n , k] * self.K[k][idx,:][:,idx]
                    # Solve the optimization problem
                    solver = SMOSolver(X_c, self.C, K, 1e-4)
                    support, _, _, _ = solver(y_c)
                     
                    # update the coefficients using the group lasso
                    for m in range(len(self.kernels)):
                        lambdas[n, m] = lambdas[n, m]**2 * (support.T @ self.K[m][idx,:][:,idx] @ support)
                    # normalization
                    lambdas[n] /= np.sum(lambdas[n])
                    n += 1

        self.lambdas = lambdas.mean(axis=0)
        print("lambdas: ", self.lambdas)
        return Kernel(kernel=self.kernels, kernel_coeff=self.lambdas)


