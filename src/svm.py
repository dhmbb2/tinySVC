import sys
sys.path.append("src/smo_kernal/build/")
import numpy as np
import ccsmo
import time
from multiprocessing import Pool

class SMOSolver:
    '''
    SMOSolver is a class that implements the Sequential Minimal Optimization algorithm for training a SVM.
    Reference: 
    [1] Platt, John. Fast Training of Support Vector Machines using Sequential Minimal Optimization, in Advances in Kernel Methods - Support Vector Learning, B. Scholkopf, C. Burges, A. Smola, eds., MIT Press (1998)
    [2] https://cs229.stanford.edu/materials/smo.pdf
    '''
    def __init__(self, X, C, kernal, tol, max_passes=1000, gamma=0.3, degree=3, lang='c++', heu=True):
        '''Args:
        C: float, regularization parameter
        tol: float, tolerance
        max_passes: int, maximum number of passes
        kernal: str, kernal function, default is None
        '''
        self.X = X
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernal_type = kernal
        self.num_sample = self.X.shape[0]
        if self.kernal_type == 'linear':
            self.kernal = self.linear_kernal
        elif self.kernal_type == 'guassian':
            self.kernal = lambda x1, x2: self.guassian_kernal(x1, x2, gamma=gamma)
        elif self.kernal_type == 'polynomial':
            self.kernal = lambda x1, x2: self.polynomial_kernal(x1, x2, degree=degree)
        else:
            raise ValueError('Invalid kernal type. Please choose from linear, guassian or polynomial')
        self.K = self.kernal(self.X, self.X)
        self.lang = lang
        assert self.lang in ['c++', 'python']
        self.heu = heu

    def update_state(self, y):
        self.y = y
        self.alphas = np.zeros(self.num_sample)
        self.b = 0

    def solve_python(self, y):
        self.update_state(y)
        passes = 0
        while(passes < self.max_passes):
            num_changed_alphas = 0
            for i in range(self.num_sample):
                Ei = self.calculate_E(i)
                if not (((self.y[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                        ((self.y[i] * Ei > self.tol) and (self.alphas[i] > 0))):
                    continue
                j = self.get_j(i)
                Ej = self.calculate_E(j)
                alpha_i_old = self.alphas[i]
                alpha_j_old = self.alphas[j]
                if self.y[i] != self.y[j]:
                    L = max(0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0, alpha_j_old + alpha_i_old - self.C)
                    H = min(self.C, alpha_j_old + alpha_i_old)
                if L == H:
                    continue
                eta = 2 * self.K[i,j] - self.K[i,i] - self.K[j,j]
                if eta >= 0: 
                    continue
                alpha_j_new = alpha_j_old - self.y[j] * (Ei - Ej) / eta
                alpha_j_new = min(H, max(L, alpha_j_new))

                if abs(alpha_j_new - alpha_j_old) < 1e-5:
                    continue
                alpha_i_new = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - alpha_j_new)
                b1 = self.b - Ei - self.y[i] * (alpha_i_new - alpha_i_old) * self.K[i,i] - self.y[j] * (alpha_j_new - alpha_j_old) * self.K[i, j]
                b2 = self.b - Ej - self.y[i] * (alpha_i_new - alpha_i_old) * self.K[i,j] - self.y[j] * (alpha_j_new - alpha_j_old) * self.K[j, j]

                if 0 < alpha_i_new < self.C:
                    b = b1
                elif 0 < alpha_j_new < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                
                self.b = b
                self.alphas[i] = alpha_i_new
                self.alphas[j] = alpha_j_new
                num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        support_idx = np.where(self.alphas > 1e-6)[0]

        # print(self.alphas)
    
        return (self.alphas * self.y)[support_idx], self.X[support_idx], self.b

    def solve(self, y):
        if self.lang == 'python':
            return self.solve_python(y)
        support, sv, b = ccsmo.solve(self.X, y.reshape(-1,1), self.kernal_type, self.C, self.tol, self.max_passes, self.heu)
        return support.reshape(-1), [sv], [np.array([b])]

    def pred(self, support, X, X_test, b):
        return support @ self.kernal(X, X_test) + b
    
    def calculate_E(self, i):
        return (self.alphas * self.y) @ self.K[:, i] + self.b - self.y[i]

    def get_j(self, i):
        return np.random.choice(np.delete(np.arange(self.num_sample), i))
    
    def linear_kernal(self, x1, x2, b=0):
        return x1 @ x2.T + b
    
    def guassian_kernal(self, x1, x2, gamma=0.3):
        num_x2_samples = x2.shape[0]
        ret = []
        for i in range(num_x2_samples):
            ret.append(np.exp(-gamma * np.linalg.norm(x1 - x2[i], axis=1)**2))
        return np.stack(ret, axis=1)
    
    def polynomial_kernal(self, x1, x2, degree=3):
        return (x1 @ x2.T + 1)**degree

class SVC:
    '''
    Args:
        X: numpy array, shape (n_samples, n_features), training data
        y: numpy array, shape (n_samples,), training labels'''
    def __init__(self, C=1, kernal='linear', tol=1e-3, max_passes=1000, gamma=0.3, degree=3, lang='c++', heu=True, threading=True):
        self.C = C
        self.kernal = kernal
        self.tol = tol
        self.max_passes = max_passes
        self.classes = None
        self.intercepts = None
        self.support_vectors = None
        self.supports = None
        self.solver = None
        self.lang = lang
        self.gamma = gamma
        self.degree = degree
        self.heu = heu
        self.threading = threading
        if self.threading:
            assert self.lang == 'c++', 'Threading is only supported for c++ implementation'

    def predict(self, X):
        y_scores = []
        if len(self.classes) == 2:
            y_scores = self.solver.pred(self.supports, self.support_vectors, X, self.intercepts)
            return np.sign(y_scores)
        for i in range(len(self.classes)):
            y_scores.append(self.solver.pred(self.supports[i], self.support_vectors[i], X, self.intercepts[i]))
        y_scores = np.array(y_scores)
        return self.classes[np.argmax(y_scores, axis=0)]

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.solver = SMOSolver(X, self.C, self.kernal, self.tol, self.max_passes, self.gamma, self.degree, self.lang, self.heu)
        if len(self.classes) == 2:
            self.supports, self.support_vectors, self.intercepts= self.solver.solve(y)
            return
        ys = []
        res = []
        if not self.threading:
            for i, c in enumerate(self.classes):
                y_c = np.where(y == c, 1, -1)
                res.append(self.solver.solve(y_c))
        else:
            pool = Pool(processes=4)
            for i, c in enumerate(self.classes):
                y_c = np.where(y == c, 1, -1)
                ys.append(y_c)
            res = pool.map(self.solver.solve, ys)
        self.supports = [k[0] for k in res]
        self.support_vectors = [k[1] for k in res]
        self.intercepts = [k[2] for k in res]

    def get_intercept(self):
        self.check_fit()
        return self.intercepts

    def get_coefs(self):
        self.check_fit()
        
        if len(self.classes) == 2:
            return self.supports @ self.support_vectors
        
        coefs = []
        for i in range(len(self.classes)):
            coefs.append(self.supports[i] @ self.support_vectors[i])
        return coefs

    def check_fit(self):
        if self.intercepts is None:
            raise ValueError('Model has not been trained yet. Please call fit method first.')
        
    
