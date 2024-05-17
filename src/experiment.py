from kernel import Kernel
from svm import SVC
import numpy as np
from classification import test_mnist, test_hog
import matplotlib.pyplot as plt
import pickle
import os
import time
from mkl import MKL
from data_preprocessing import load_cifar10_hog, load_mnist_data


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def draw(X_train, y_train, svc, name):
    # Fit the model
    start = time.time()
    svc.fit(X_train, y_train)
    print(f"Time: {time.time()-start}")
    path = f"exp_imgs/{name}.png"

    # Create a grid to evaluate model
    xx = np.linspace(0, np.max(X_train), 100)
    # w, b = svc.coef_[0], svc.intercept_
    # ww, bb = svc.get_coefs(), svc.get_intercept()

    plt.figure()
    # X_test = np.random.normal(loc=[0,0], scale=[4,4], size=(100, 2))
    # y_pred = svc.predict(X_test)

    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
    plot_contours(plt, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot decision boundary
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='Training data')
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', label='Test data')
    plt.legend()
    plt.savefig(path)

def abla_linear():
    path = 'exp_imgs/abla_linear.png'
    pkl_path = 'exp_output/abla_linear.pkl'
    ts = np.arange(-7, 2, 1)
    plt.clf()
    results = []
    for t in ts:
        acc = test_mnist(C=pow(2.0, t))
        print(f"Accuracy: {acc} for C=2^{t}")
        results.append(acc)

    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    plt.plot(ts, results, '--o', color='darkcyan')
    plt.xlabel("C")
    plt.ylabel('Accuracy')
    plt.title("ablation study of linear kernel")
    plt.grid(visible=0.5, axis='y')
    plt.xticks(ts, [f'2^{t}' for t in ts])
    plt.savefig(path)

def abla_poly():
    path = 'exp_imgs/abla_poly.png'
    ts = np.arange(-7, 1, 1)
    ds = np.arange(1, 5, 1)
    colors = ['darkcyan', 'khaki', 'mediumseagreen', 'mediumpurple']
    plt.clf()
    all_results = []
    for i in range(len(ds)):
        results = []
        for t in ts:
            acc = test_mnist(kernel='polynomial', gamma_poly=2.0**t, degree=ds[i])
            print(f"Accuracy: {acc} for gamma_poly={t} and degree={ds[i]}")
            results.append(acc)
        all_results.append(results)
        plt.plot(ts, results, '--o', color=colors[i], label=f'degree={ds[i]}')
    
    pickle.dump(all_results, open('exp_output/abla_poly.pkl', 'wb'))
    plt.legend()
    plt.xlabel("gamma")
    plt.ylabel('Accuracy')
    plt.title("ablation study of polynomial kernel")
    plt.grid(visible=0.5, axis='y')
    plt.xticks(ts)
    plt.savefig(path)


def abla_guassian():
    path = 'exp_imgs/abla_guassian.png'
    ts = np.arange(-7, 1, 1)
    Cs = np.arange(-3, 1, 1)
    colors = ['darkcyan', 'khaki', 'mediumseagreen', 'mediumpurple']
    plt.clf()
    results = []
    all_results = []
    for i in range(len(Cs)):
        results = []
        for t in ts:
            acc = test_mnist(kernel='guassian', gamma=2.0**t, C=2.0**Cs[i])
            print(f"Accuracy: {acc} for gamma_poly={t}, C=2^{Cs[i]}")
            results.append(acc)
        all_results.append(results)
        plt.plot(ts, results, '--o', color=colors[i], label=f'C={Cs[i]}')

    with open('exp_output/abla_guassian.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    plt.legend()
    plt.xlabel("gamma")
    plt.ylabel('Accuracy')
    plt.title("ablation study of guassian kernel")
    plt.grid(visible=0.5, axis='y')
    plt.xticks(ts)
    plt.savefig(path)

def plot_results():
    with open('exp_output/abla_linear.pkl', 'rb') as f:
        all_results = pickle.load(f)
    ts = np.arange(-7, 2, 1)
    # ds = np.arange(1,5,1)
    colors = ['darkcyan', 'khaki', 'mediumseagreen', 'mediumpurple']
    # for i, results in enumerate(all_results):
    plt.plot(ts, all_results, '--o', color=colors[0])
    # plt.legend()
    plt.xlabel("C")
    plt.ylabel('Accuracy')
    plt.title("ablation study of linear kernel")
    plt.grid(visible=0.5, axis='y')
    plt.xticks(ts)
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.savefig('exp_imgs/abla_poly2.png')

def test_heu():
    
    num = np.arange(50, 500, 50)
    times1 = []
    times2 = []
    times3 = []
    times4 = []
    for n in num:
        X1 = np.random.normal(loc=-1, scale=1, size=(n, 10))
        X2 = np.random.normal(loc=1, scale=1, size=(n, 10))
        X_train = np.concatenate([X1, X2])

        y1 = -1 * np.ones(n)
        y2 = 1 * np.ones(n)
        y_train = np.concatenate([y1, y2])

        svc1 = SVC(kernel='linear', max_passes=100, heu=True, lang='python')
        svc2 = SVC(kernel='linear', max_passes=100, heu=False, lang='python')
        svc3 = SVC(kernel='linear', max_passes=100, heu=True, lang='c++')
        svc4 = SVC(kernel='linear', max_passes=100, heu=False, lang='c++')

        t1 = time.time()
        svc1.fit(X_train, y_train)
        t2  = time.time()
        svc2.fit(X_train, y_train)
        t3 = time.time()
        svc3.fit(X_train, y_train)
        t4 = time.time()
        svc4.fit(X_train, y_train)
        t5 = time.time()

        print(f"{n}: Python with heu: {t2-t1}")
        print(f"{n}: Python without heu: {t3-t2}")
        print(f"{n}: Cpp with heu: {t4-t3}")
        print(f"{n}: Cpp without heu: {t5-t4}")

        times1.append(t2-t1)
        times2.append(t3-t2)
        times3.append(t4-t3)
        times4.append(t5-t4)

    plt.clf()
    plt.plot(num, times1, '--o', label='Python with heu')
    plt.plot(num, times3, '--o', label='Cpp with heu')
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel('Time')
    plt.title("Time comparison with heu")
    plt.grid(visible=0.5, axis='y')
    plt.savefig('exp_imgs/time_heu.png')

    plt.clf()
    plt.plot(num, times2, '--o', label='Python without heu')
    plt.plot(num, times4, '--o', label='Cpp without heu')
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel('Time')
    plt.title("Time comparison without heu")
    plt.grid(visible=0.5, axis='y')
    plt.savefig('exp_imgs/time_no_heu.png')

def draw_multiclass():
    X1 = np.random.normal(loc=[-5, -5], scale=1, size=(500, 2))
    X2 = np.random.normal(loc=[6, 6], scale=1, size=(500, 2))
    X3 = np.random.normal(loc=[10, -10], scale=[1, 1], size=(500, 2))
    X4 = np.random.normal(loc=[-6, 10], scale=[1, 1], size=(500, 2))
    X_train = np.concatenate([X1, X2, X3, X4])

    y1 = 1 * np.ones(500)
    y2 = 2 * np.ones(500)
    y3 = 3 * np.ones(500)
    y4 = 4 * np.ones(500)
    y_train = np.concatenate([y1, y2, y3, y4])

    svc1 = SVC(max_passes=100, heu=True, lang='python', strategy='ovo')
    svc2 = SVC(max_passes=100, heu=True, lang='c++', strategy='ovr')
    draw(X_train, y_train, svc1, 'ovo')
    draw(X_train, y_train, svc2, 'ovr')

def test_multiclass():
    t1 = time.time()
    ovo_acc, ovo_train_acc = test_mnist(C=1, strategy='ovo')
    t2 = time.time()
    ovr_acc, ovr_train_acc = test_mnist(C=1, strategy='ovr')
    t3 = time.time()

    print(f"OVO: {ovo_acc} and train_acc={ovo_train_acc}")
    print(f"OVR: {ovr_acc} and train_acc={ovr_train_acc}")
    print(f"OVO time: {t2-t1}")
    print(f"OVR time: {t3-t2}")


def test_boundry():
    X1 = np.random.normal(loc=0, scale=1, size=(50, 2))

    angles = np.linspace(0, np.pi, 50, endpoint=False)
    out_r = np.random.uniform(4-1/2 , 4+1/2, 50)
    outer_points_x = np.cos(angles) * out_r
    outer_points_y = np.sin(angles) * out_r
    X2 = np.stack([outer_points_x, outer_points_y], axis=-1)

    angles = np.linspace(np.pi, 2 * np.pi, 50, endpoint=False)
    out_r = np.random.uniform(4-1/2 , 4+1/2, 50)
    outer_points_x = np.cos(angles) * out_r
    outer_points_y = np.sin(angles) * out_r
    X3 = np.stack([outer_points_x, outer_points_y], axis=-1)

    X_train = np.concatenate([X1, X2, X3])
    y_train = np.concatenate([np.ones(50), 2 * np.ones(50), 3 * np.ones(50)])

    svc1 = SVC(C=1, kernel='guassian', gamma=0.3)
    svc2 = SVC(C=1, kernel='linear')
    svc3 = SVC(C=1, kernel='polynomial', degree=2)

    draw(X_train, y_train, svc1, 'guassian')
    draw(X_train, y_train, svc2, 'linear')
    draw(X_train, y_train, svc3, 'polynomial')

def test_mkl_mnist():
    image_train, label_train, image_test, label_test = load_mnist_data()
    image_train = image_train.reshape(-1, 28*28)
    image_test = image_test.reshape(-1, 28*28)
    rbf1 = Kernel(kernel='rbf', gamma_gua=2.0 ** -4)
    rbf2 = Kernel(kernel='rbf', gamma_gua=2.0 ** -5)
    rbf3 = Kernel(kernel='rbf', gamma_gua=2.0 ** -6)
 
    mkl = MKL(kernels = [rbf1, rbf2, rbf3])
    kernel = mkl.get_kernel(image_train, label_train)
    acc_mn, train_acc_mn = test_mnist(C=1, kernel=kernel)
    print(f"Accuracy: {acc_mn} and train_acc={train_acc_mn}")

    poly1 = Kernel(kernel='poly', degree=3, gamma_poly=2.0**-5)
    poly2 = Kernel(kernel='poly', degree=2, gamma_poly=2.0**-2)
    poly3 = Kernel(kernel='poly', degree=4, gamma_poly=2.0**-6)

    mkl = MKL(kernels = [poly1, poly2, poly3])
    kernel = mkl.get_kernel(image_train, label_train)
    acc_po, train_acc_po = test_mnist(kernel=kernel)
    print(f"Accuracy: {acc_po} and train_acc={train_acc_po}")

    mkl = MKL(kernels = [rbf1, rbf2, rbf3, poly1, poly2, poly3])
    kernel = mkl.get_kernel(image_train, label_train)
    acc, train_acc = test_mnist(kernel=kernel)
    print(f"Accuracy: {acc} and train_acc={train_acc}")

def test_mkl_hog():
    image_train, label_train, image_test, label_test = load_cifar10_hog()
    rbf1 = Kernel(kernel='rbf', gamma_gua=0.03)
    rbf2 = Kernel(kernel='rbf', gamma_gua=0.05)
    rbf3 = Kernel(kernel='rbf', gamma_gua=0.07)
 
    mkl = MKL(kernels = [rbf1, rbf2, rbf3])
    kernel = mkl.get_kernel(image_train, label_train)
    test_hog(kernel=kernel)

    poly1 = Kernel(kernel='poly', degree=3, gamma_poly=0.07)
    poly2 = Kernel(kernel='poly', degree=2, gamma_poly=0.1)
    poly3 = Kernel(kernel='poly', degree=4, gamma_poly=0.05)

    mkl = MKL(kernels = [poly1, poly2, poly3])
    kernel = mkl.get_kernel(image_train, label_train)
    test_hog(kernel=kernel)

    mkl = MKL(kernels = [rbf1, rbf2, rbf3, poly1, poly2, poly3])
    kernel = mkl.get_kernel(image_train, label_train)
    test_hog(kernel=kernel)


def hog_acc():
    acc1, train_acc1 = test_hog(C=0.7, kernel='linear')
    acc2, train_acc2 = test_hog(C=0.1, kernel='polynomial', degree=2, gamma_poly=0.1)
    acc3, train_acc3 = test_hog(C=0.7, kernel='guassian', gamma=0.1)
    print(f"Linear: {acc1} and train_acc={train_acc1}")
    print(f"Poly: {acc2} and train_acc={train_acc2}")
    print(f"Guassian: {acc3} and train_acc={train_acc3}")

def mnist_acc():
    acc1, train_acc1 = test_mnist(C=2**-5, kernel='linear')
    acc2, train_acc2 = test_mnist(C=1e-2, kernel='polynomial', degree=2, gamma_poly=2.0**-2)
    acc3, train_acc3 = test_mnist(C=1, kernel='guassian', gamma=2.0**-5)
    print(f"Linear: {acc1} and train_acc={train_acc1}")
    print(f"Poly: {acc2} and train_acc={train_acc2}")
    print(f"Guassian: {acc3} and train_acc={train_acc3}")

def abla_mkl():
    image_train, label_train, image_test, label_test = load_cifar10_hog()
    rbf1 = Kernel(kernel='rbf', gamma_gua=0.05)
    rbf2 = Kernel(kernel='rbf', gamma_gua=0.03)
    rbf3 = Kernel(kernel='rbf', gamma_gua=0.07)
 
    kernel = Kernel(kernel=[rbf1, rbf2, rbf3], kernel_coeff=[0.333,0.333,0.333])
    test_hog(kernel=kernel)

    poly1 = Kernel(kernel='poly', degree=3, gamma_poly=0.07)
    poly2 = Kernel(kernel='poly', degree=2, gamma_poly=0.1)
    poly3 = Kernel(kernel='poly', degree=4, gamma_poly=0.05)

    kernel = Kernel(kernel=[poly1, poly2, poly3], kernel_coeff=[0.333,0.333,0.333])
    test_hog(kernel=kernel)

    kernel = Kernel(kernel=[rbf1, rbf2, rbf3, poly1, poly2, poly3], kernel_coeff=[0.166,0.166,0.166,0.166,0.166,0.166])
    test_hog(kernel=kernel)

    image_train, label_train, image_test, label_test = load_mnist_data()
    image_train = image_train.reshape(-1, 28*28)
    image_test = image_test.reshape(-1, 28*28)
    rbf1 = Kernel(kernel='rbf', gamma_gua=2.0 ** -4)
    rbf2 = Kernel(kernel='rbf', gamma_gua=2.0 ** -5)
    rbf3 = Kernel(kernel='rbf', gamma_gua=2.0 ** -6)
    
    kernel = Kernel(kernel=[rbf1, rbf2, rbf3], kernel_coeff=[0.333,0.333,0.333])
    test_mnist(kernel=kernel)

    poly1 = Kernel(kernel='poly', degree=3, gamma_poly=2.0**-5)
    poly2 = Kernel(kernel='poly', degree=2, gamma_poly=2.0**-2)
    poly3 = Kernel(kernel='poly', degree=4, gamma_poly=2.0**-6)

    kernel = Kernel(kernel=[poly1, poly2, poly3], kernel_coeff=[0.333,0.333,0.333])
    test_mnist(kernel=kernel)

    kernel = Kernel(kernel=[rbf1, rbf2, rbf3, poly1, poly2, poly3], kernel_coeff=[0.166,0.166,0.166,0.166,0.166,0.166])
    test_mnist(kernel=kernel)