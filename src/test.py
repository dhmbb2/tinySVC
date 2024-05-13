import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.svm import SVC
from sklearn.svm import SVC as skSVC
import sys
import time
from SVM import SupportVectorClassifier

# Create a SVC instance

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
    path = f"test_img/2d_decision_boundary_with_test_data_{name}.png"

    # Create a grid to evaluate model
    xx = np.linspace(0, np.max(X_train), 100)
    # w, b = svc.coef_[0], svc.intercept_
    # ww, bb = svc.get_coefs(), svc.get_intercept()

    plt.figure()
    # # Calculate decision boundary (z = wx + b)
    # for i in range(4):
    #     zz = (-ww[i][0, 0]/ww[i][0, 1]) * xx - bb[i]/ww[i][0, 1]
    #     plt.plot(xx, zz[0])
    # # zz = (-ww[0, 0]/ww[0, 1]) * xx - bb/ww[0, 1]
    # plt.plot(xx, zz[0])
    # #创建更多的测试数据
    X_test = np.random.normal(loc=[0,0], scale=[4,4], size=(100, 2))
    #使用模型进行预测
    y_pred = svc.predict(X_test)

    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
    plot_contours(plt, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot decision boundary
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='Training data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', label='Test data')
    plt.legend()
    plt.savefig(path)

def test_linear():
    X1 = np.random.normal(loc=-5, scale=1, size=(500, 2))
    X2 = np.random.normal(loc=6, scale=1, size=(500, 2))
    # X3 = np.random.normal(loc=[10, -10], scale=[1, 1], size=(5000, 2))
    # X4 = np.random.normal(loc=[-6, 10], scale=[1, 1], size=(500, 2))
    X_train = np.concatenate([X1, X2])

    y1 = -1 * np.ones(500)
    y2 = 1 * np.ones(500)
    # y3 = 1 * np.ones(5000)
    # y4 = 1 * np.ones(500)
    y_train = np.concatenate([y1, y2])

    # svc_gt = LinearSVC()
    svc1 = SVC(kernal='linear', max_passes=100, threading=False, heu=True, lang='python')
    # svc1 = skSVC(kernel='linear')
    # svc1 = SupportVectorClassifier()
    # t1 = time.time()
    # svc1.fit(X_train, y_train)
    # t2  = time.time()
    # print(f"Time: {t2-t1}")
    draw(X_train, y_train, svc1, 'python_heu')

def test_kernal():
    num_point = 50
    eps = 3 

    inner_points_x = np.random.randn(num_point) * eps
    inner_points_y = np.random.randn(num_point) * eps
    X1 = np.stack([inner_points_x, inner_points_y], axis=-1)
    Y1 = np.ones(num_point)

    outer_radius = 8

    angles = np.linspace(0, 2 * np.pi, num_point, endpoint=False)
    out_r = np.random.uniform(outer_radius-eps/2 , outer_radius+eps/2, num_point)
    outer_points_x = np.cos(angles) * out_r
    outer_points_y = np.sin(angles) * out_r
    X2 = np.stack([outer_points_x, outer_points_y], axis=-1)
    Y2 = 2 * np.ones(num_point)

    out_outer_radius = 12
    angles = np.linspace(0, 2 * np.pi, num_point, endpoint=False)
    out_r = np.random.uniform(out_outer_radius-eps/2 , out_outer_radius+eps/2, num_point)
    outer_points_x = np.cos(angles) * out_r
    outer_points_y = np.sin(angles) * out_r
    X3 = np.stack([outer_points_x, outer_points_y], axis=-1)
    Y3 = 3 * np.ones(num_point)

    svc = SVC(kernal='guassian', gamma=0.3)

    X = np.concatenate([X1, X2, X3])
    Y = np.concatenate([Y1, Y2, Y3])

    draw(X, Y, svc, 'guassian')

def test_cpp_kernal():
    X1 = np.random.normal(loc=-1, scale=1, size=(50, 2))
    X2 = np.random.normal(loc=2, scale=1, size=(50, 2))

    X_train = np.concatenate([X1, X2])
    y_train = np.concatenate([np.ones(50), -np.ones(50)])

    svc_gt = skSVC(kernel='linear')
    svc1 = SVC(lang='python', kernal='linear')
    svc2 = SVC(lang='c++', kernal='linear')

    svc_gt.fit(X_train, y_train)
    svc1.fit(X_train, y_train)
    svc2.fit(X_train, y_train)

def measure_time():
    X1 = np.random.normal(loc=-5, scale= 1, size=(500, 2))
    X2 = np.random.normal(loc=6, scale=1, size=(500, 2))
    X3 = np.random.normal(loc=10, scale=1, size=(500, 2))
    X4 = np.random.normal(loc=-10, scale=1, size=(500, 2))
    X_train = np.concatenate([X1, X2, X3, X4])

    y1 = np.ones(500)
    y2 = -1 * np.ones(500)
    y3 =  -1 * np.ones(500)
    y4 = -1  * np.ones(500)
    y_train = np.concatenate([y1, y2, y3, y4])

    svcgt = skSVC(kernel='linear')
    svc_cpp_multi= SVC(kernal='linear', max_passes=100, heu=True)
    svc_cpp_single = SVC(kernal='linear', max_passes=100, heu=True, threading=False)
    svc_other = SupportVectorClassifier()
    svc_python_heu = SVC(kernal='linear', lang='python', threading=False)
    # svc_cpp_withoutheu = SVC(kernal='linear', max_passes=100, heu=False)
    # svc_python = SVC(kernal='linear', lang='python', threading=False)

    t4 = time.time()
    svc_other.fit(X_train, y_train)
    t5 = time.time()
    svc_python_heu.fit(X_train, y_train)
    t6 = time.time()
    # svc_cpp_withoutheu.fit(X_train, y_train)
    # t4 = time.time()
    # svc_python.fit(X_train, y_train)
    # t5 = time.time()

    print("Fitting time:")
    # print(f"cpp multithread: {t3-t2}")
    # print(f"cpp singlethread: {t4-t3}")
    print(f"other: {t5-t4}")
    print(f"python heu: {t6-t5}")
    # print(f"cpp without heu: {t4-t3}")
    # print(f"python: {t5-t4}")


# measure_time()
test_linear()
# test_kernal()