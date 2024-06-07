import sys
import os
if os.path.exists("src/smo_kernel/build/"):
    sys.path.append("src/smo_kernel/build/")
if os.path.exists("lib/"):
    sys.path.append("lib/")
import numpy as np
import ccsmo
import time
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from kernel import Kernel
import pickle

class SMOSolver:
    def __init__(self, X, C, kernel, tol=1e-4, max_passes=100, gamma=0.3, degree=3, lang='python', heu=True):
        """SMOSolver class to solve the optimization problem in SVM
        Args:
            X: np.array, training data
            C: float, regularization parameter
            kernel: np.array, kernel matrix
            tol: float, tolerance
            max_passes: int, maximum number of passes
            gamma: float, gamma for gaussian kernel
            degree: int, degree for polynomial kernel
            lang: string, language to use, either 'c++' or 'python'
            heu: bool, whether to use heuristics to choose the second alpha
        """
        self.X = X
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.num_sample = self.X.shape[0]
        self.gamma = gamma
        self.degree = degree
        self.K = kernel
        self.lang = lang
        assert self.lang in ['c++', 'python']
        self.heu = heu
        self.E_cache = np.zeros((self.num_sample))

    def update_state(self, y):
        self.y = y
        self.alphas = np.zeros(self.num_sample)
        self.b = 0

    def j_loop(self, i):
        # inner loop for smo algorithm
        Ei = self.calculate_E(i)
        if ((self.y[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                ((self.y[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j = self.get_j(Ei, i)
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
                return 0
            eta = 2 * self.K[i,j] - self.K[i,i] - self.K[j,j]
            if eta >= 0: 
                return 0
            alpha_j_new = np.clip(alpha_j_old - self.y[j] * (Ei - Ej) / eta, L, H)

            if abs(alpha_j_new - alpha_j_old) < self.tol:
                return 0
            alpha_i_new = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - alpha_j_new)
            b1 = self.b - Ei - self.y[i] * (alpha_i_new - alpha_i_old) * self.K[i,i] - self.y[j] * (alpha_j_new - alpha_j_old) * self.K[i, j]
            b2 = self.b - Ej - self.y[i] * (alpha_i_new - alpha_i_old) * self.K[i,j] - self.y[j] * (alpha_j_new - alpha_j_old) * self.K[j, j]

            if 0 < alpha_i_new < self.C:
                b = b1
            elif 0 < alpha_j_new < self.C:
                b = b2
            else:
                b = (b1 + b2) / 2
                
            #update bounding E_cache
            valid_idx = np.where((self.alphas > self.tol) * (self.alphas < self.C - self.tol))[0] 
            self.E_cache[valid_idx] += self.y[valid_idx] * (alpha_i_new - alpha_i_old) * self.K[valid_idx, i] \
                                + self.y[j] * (alpha_j_new - alpha_j_old) * self.K[valid_idx, j] + self.b - b

            self.b = b
            self.alphas[i] = alpha_i_new
            self.alphas[j] = alpha_j_new
            self.E_cache[i] = 0
            self.E_cache[j] = 0
            return 1
        return 0
    
    def solve_python_heu(self, y):
        passes = 0
        num_changed_alphas = 0
        iter_whole_set = 1

        # outer loop: first iterate over the whole set, then iterate over the non-boundary alphas
        # if the non-boundary alphas do not change, iterate over the whole set again. Until convergence
        while passes < self.max_passes and (num_changed_alphas or iter_whole_set):
            num_changed_alphas = 0
            if iter_whole_set:
                for i in range(self.num_sample):
                    num_changed_alphas += self.j_loop(i)
                passes += 1
            else:
                index = np.nonzero((self.tol < self.alphas) * (self.alphas < self.C - self.tol))[0]
                for i in index:
                    num_changed_alphas += self.j_loop(i)
                passes += 1
            if iter_whole_set:
                iter_whole_set = 0
            elif num_changed_alphas == 0:
                iter_whole_set = 1
        support_idx = np.where(self.alphas > self.tol)[0]
        return (self.alphas * self.y), (self.alphas * self.y)[support_idx], self.X[support_idx], self.b

    def solve_python_simple(self, y):
        self.update_state(y)
        passes = 0
        while(passes < self.max_passes):
            num_changed_alphas = 0
            for i in range(self.num_sample):
                num_changed_alphas += self.j_loop(i)

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        support_idx = np.where(self.alphas > self.tol)[0]
        return (self.alphas * self.y)[support_idx], self.X[support_idx], self.b

    def solve_python(self, y):
        self.update_state(y)
        if self.heu:
            return self.solve_python_heu(y)
        return self.solve_python_simple(y)

    def __call__(self, y):
        if self.lang == 'python':
            return self.solve_python(y)
        support, sv, b = ccsmo.solve(self.X, y.reshape(-1,1), self.K, self.C, self.tol, self.max_passes, self.heu)
        return support.reshape(-1), sv, np.array([np.array([b])])
    
    def calculate_E(self, i):
        return (self.alphas * self.y) @ self.K[:, i] + self.b - self.y[i]

    def get_j(self, Ei, i):
        if not self.heu:
            return np.random.choice(np.delete(np.arange(self.num_sample), i))
        # if using heuristics, choosing the j that maximizes the |Ei - Ej|
        valid_idx = np.where((self.alphas > self.tol) * (self.alphas < self.C - self.tol))[0]
        if len(valid_idx) > 0:
            j = valid_idx[np.argmax(np.abs(self.E_cache[valid_idx] - Ei))]
        else:
            j = np.random.choice(np.delete(np.arange(self.num_sample), i))
        return j
        
class SVC:
    def __init__(self, C=1, kernel='linear', tol=1e-5, max_passes=100, gamma=0.03, gamma_poly=1, degree=3, lang='python', heu=True, strategy='ovo'):
        """Support Vector Classifier class
        Args:
            C: float, regularization parameter
            kernel: string, kernel type
            tol: float, tolerance
            max_passes: int, maximum number of passes
            gamma: float, gamma for gaussian kernel
            gamma_poly: float, gamma for polynomial kernel
            degree: int, degree for polynomial kernel
            lang: string, language to use, either 'c++' or 'python'
            heu: bool, whether to use heuristics to choose the second alpha
            strategy: string, one-vs-one or one-vs-rest
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.classes = None
        self.intercepts = None
        self.support_vectors = None
        self.supports = None
        self.classes_pair = []
        self.lang = lang
        self.gamma = gamma
        self.degree = degree
        self.heu = heu
        self.num_classes = None
        self.strategy = strategy
        assert self.strategy in ['ovr', 'ovo'], 'Invalid strategy. Please choose from ovr or ovo'
        self.gamma_poly = gamma_poly

        if isinstance(kernel, Kernel):
            self.kernel = kernel
        else:
            if kernel == 'linear':
                self.kernel = self.linear_kernel
            elif kernel == 'gaussian':
                self.kernel = self.gaussian_kernel
            elif kernel == 'polynomial':
                self.kernel = self.polynomial_kernel
            else:
                raise ValueError('Invalid kernel type. Please choose from linear, gaussian or polynomial or custom')


    def fit(self, X, y, ):
        self.K = self.kernel(X, X)
        self.classes = np.unique(y)
        if self.strategy == 'ovr':
            self.ovr_fit(X, y)
        elif self.strategy == 'ovo':
            self.ovo_fit(X, y)

    def predict(self, X):
        if self.strategy == 'ovr':
            return self.predict_ovr(X)
        return self.predict_ovo(X)

    def ovr_fit(self, X, y):
        self.solver = SMOSolver(X, self.C, self.K, self.tol, self.max_passes, self.gamma, self.degree, self.lang, self.heu)
        if len(self.classes) == 2:
            _, self.supports, self.support_vectors, self.intercepts= self.solver(y)
            return
        res = []
        for i, c in tqdm(enumerate(self.classes)):
            print(f'Fitting for class {c}')
            y_c = np.where(y == c, 1, -1)
            res.append(self.solver(y_c))
        self.supports = [k[0] for k in res]
        self.support_vectors = [k[1] for k in res]
        self.intercepts = [k[2] for k in res]

    def ovo_fit(self, X, y):
        self.supports = []
        self.support_vectors = []
        self.intercepts = []
        with tqdm(total=len(self.classes) * (len(self.classes) - 1) // 2) as pbar:
            for i in range(len(self.classes)):
                for j in range(i+1, len(self.classes)):
                    # get the indices of the classes
                    idx = np.where((y == self.classes[i]) | (y == self.classes[j]))[0]
                    X_c= X[idx]
                    y_c = np.where(y[idx] == self.classes[i], -1, 1)
                    solver = SMOSolver(X_c, self.C, self.K[idx,:][:,idx], self.tol, self.max_passes, self.gamma, self.degree, self.lang, self.heu)
                    _, support, sv, b = solver(y_c)
                    self.classes_pair.append((i, j))
                    self.supports.append(support)
                    self.support_vectors.append(sv)
                    self.intercepts.append(b)
                    pbar.update(1)
                # print(f'Finish training for class {self.classes[i]} and {self.classes[j]}')

    def predict_ovr(self, X):
        y_scores = []
        if len(self.classes) == 2:
            y_scores = self.supports @ self.kernel(self.support_vectors, X) + self.intercepts
            return np.sign(y_scores)
        for i in range(len(self.classes)):
            y_scores.append(self.supports[i] @ self.kernel(self.support_vectors[i], X) + self.intercepts[i])
        y_scores = np.array(y_scores)
        return self.classes[np.argmax(y_scores, axis=0)]

    def predict_ovo(self, X):
        vote = np.zeros((X.shape[0], len(self.classes)))
        # voting for each pair of classes
        for idx, (i, j) in tqdm(enumerate(self.classes_pair)):
            y = np.sign(self.supports[idx] @ self.kernel(self.support_vectors[idx], X) + self.intercepts[idx])
            vote[np.where(y == -1)[0], i] += 1
            vote[np.where(y == 1)[0], j] += 1
        return self.classes[np.argmax(vote, axis=1)]

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
        
    def linear_kernel(self, x1, x2):
        return x1 @ x2.T
    
    def gaussian_kernel(self, x1, x2):
        num_x2_samples = x2.shape[0]
        ret = []
        for i in range(num_x2_samples):
            ret.append(np.exp(-self.gamma * np.linalg.norm(x1 - x2[i], axis=1)**2))
        ret = np.stack(ret, axis=1)
        return ret
    
    def polynomial_kernel(self, x1, x2):
        return (self.gamma_poly * x1 @ x2.T + 1)**self.degree