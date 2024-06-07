import numpy as np
import pickle
import os

class Kernel:
    def __init__(self, kernel='linear', degree=2, gamma_gua=0.03, gamma_sig=1, gamma_poly=0.1, coef0=0, kernel_coeff=[1.0]):
        """Kernel class to define different types of kernels, support multiple kernels combination 
        with customized coefficients
        Args:
            kernel: string or list of strings or list of Kernel objects, kernel type(s) to be used
            degree: int, degree of polynomial kernel
            gamma_gua: float, gamma for gaussian kernel
            gamma_sig: float, gamma for sigmoid kernel
            gamma_poly: float, gamma for polynomial kernel
            coef0: float, constant term for polynomial and sigmoid kernel
            kernel_coeff: list of floats, coefficients for each kernel
        """
        self.kernel_type = kernel
        self.degree = degree
        self.gamma_gua = gamma_gua
        self.gamma_sig = gamma_sig
        self.gamma_poly = gamma_poly
        self.kernel_coeff = kernel_coeff
        self.coef0 = coef0
        if isinstance(kernel, str):
            assert kernel in ['linear', 'poly', 'rbf'], f"Kernel {kernel} not implemented"
            kernel = [kernel]
        elif isinstance(kernel[0], str): 
            assert len(kernel) == len(kernel_coeff), "Number of kernels and kernel coefficients should be the same"
            for i in kernel:
                assert i in ['linear', 'poly', 'rbf'], f"Kernel {i} not implemented"
        else:
            assert len(kernel) == len(kernel_coeff), "Number of kernels and kernel coefficients should be the same"
            for i in kernel:
                assert isinstance(i, Kernel), "Kernels passed in should either be of type Kernel or string, but not mixed"
        
        self.kernel = kernel

    def linear_kernel(self, x1, x2):
        return x1 @ x2.T
    
    def gaussian_kernel(self, x1, x2):
        num_x2_samples = x2.shape[0]
        ret = []
        for i in range(num_x2_samples):
            ret.append(np.exp(-self.gamma_gua * np.linalg.norm(x1 - x2[i], axis=1)**2))
        return np.stack(ret, axis=1)
    
    def polynomial_kernel(self, x1, x2):
        return (self.gamma_poly * x1 @ x2.T + 1)**self.degree

    def kernel_function(self, kernel, x, y):
        if isinstance(kernel, Kernel):
           return kernel(x, y)
        if kernel == 'linear':
            return self.linear_kernel(x, y)
        elif kernel == 'poly':
            return self.polynomial_kernel(x, y)
        elif kernel == 'rbf':
            return self.gaussian_kernel(x, y)
        else:
            raise ValueError(f"Kernel {kernel} not implemented")
        
    def __call__(self, x, y):
        ret = np.zeros((x.shape[0], y.shape[0]))
        for kernel, coeff in zip(self.kernel, self.kernel_coeff):
            ret += coeff * self.kernel_function(kernel, x, y)
        return ret