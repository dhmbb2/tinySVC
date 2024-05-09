import numpy as np

class SMO_SOLVER:
    '''
    SMO_SOLVER is a class that implements the Sequential Minimal Optimization algorithm for training a SVM.
    Reference: 
    [1] Platt, John. Fast Training of Support Vector Machines using Sequential Minimal Optimization, in Advances in Kernel Methods - Support Vector Learning, B. Scholkopf, C. Burges, A. Smola, eds., MIT Press (1998)
    [2] https://cs229.stanford.edu/materials/smo.pdf
    '''
    def __init__(self, X, y, C, kernal, tol, max_passes=1000):
        '''Args:
        X: numpy array, shape (n_samples, n_features), training data
        y: numpy array, shape (n_samples,), training labels
        C: float, regularization parameter
        tol: float, tolerance
        max_passes: int, maximum number of passes
        kernal: str, kernal function, default is None
        '''
        self.X = X
        self.y = y
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        if kernal is None:
            self.kernal = self.linear_kernal
        self.init_params()

    def init_params(self):
        self.num_sample = self.X.shape[0]
        self.alphas = np.zeros(self.X.shape[0])
        self.w = np.zeros(self.X.shape[1])
        self.b = 0
        self.K = self.kernal(self.X, self.X)

    def solve(self):
        passes = 0
        while(passes < self.max_passes):
            num_changed_alphas = 0
            for i in range(self.num_sample):
                Ei = self.f(i) - self.y[i]
                if not (((self.y[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                        ((self.y[i] * Ei > self.tol) and (self.alphas[i] > 0))):
                    continue
                a = np.delete(np.arange(self.num_sample), i)
                j = np.random.choice(np.delete(np.arange(self.num_sample), i))
                Ej = self.f(j) - self.y[j]
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
            print(self.alphas)


        self.cal_w()
        return self.w, self.b

    def cal_w(self):
        self.w = (self.alphas * self.y) @ self.X

    def linear_kernal(self, x1, x2, b=0):
        return x1 @ x2.T + b

    def f(self, i):
        return (self.alphas * self.y) @ self.K[:, i] + self.b

class SVC:
    def __init__(self, kernal='linear', max_iter=1000, C=1.0):
        self.kernal = kernal
        self.max_iter = max_iter
        self.C = C
        self.w = None
        self.b = None

    def param(self):
        return self.w, self.b

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError('Model is not trained yet')
        return np.sign(self.w @ X.T + self.b)

    def fit(self, X, y):
        self.smo = SMO_SOLVER(X, y, self.C, None, 1e-6, self.max_iter)
        self.w, self.b = self.smo.solve()
        
    
