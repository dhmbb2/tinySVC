import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from svm import SVC
from sklearn.svm import SVC as skSVC

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

# # Create a simple dataset
# X_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
# y_train = np.array([1, 1, -1, -1])

# # Fit the model
# svc.fit(X_train, y_train)

# # Create a grid to evaluate model
# xx, yy = np.meshgrid(range(0, 5), range(0, 5))
# w, b = svc.param()
# print(w, b)
# zz = (b + w[0]*xx + w[1]*yy) / w[2]

# # Plot decision boundary in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, zz, color='b', alpha=0.2)
# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, s=100)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.savefig('3d_decision_boundary.png')

# np.random.seed(1)

# Create a simple 2D dataset
# X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# y_train = np.array([1, 1, -1, -1])

def draw(X_train, y_train, svc):
    # Fit the model
    svc.fit(X_train, y_train)

    # Create a grid to evaluate model
    xx = np.linspace(0, np.max(X_train), 100)
    # w, b = svc.coef_[0], svc.intercept_
    ww, bb = svc.get_coefs(), svc.get_intercept()
    print(ww, bb)

    # # Calculate decision boundary (z = wx + b)
    # for i in range(3):
    #     zz = (-ww[i][0]/ww[i][1]) * xx - bb[i]/ww[i][1]
    #     plt.plot(xx, zz)
    # 创建更多的测试数据
    # X_test = np.random.normal(loc=[0,0], scale=[4,4], size=(100, 2))
    # 使用模型进行预测
    # y_pred = svc.predict(X_test)


    xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
    plot_contours(plt, svc, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot decision boundary
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='Training data')
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', label='Test data')
    plt.legend()
    plt.savefig('test_img/2d_decision_boundary_with_test_data.png')

def test_linear():
    X1 = np.random.normal(loc=[-5, -5], scale=[1, 1], size=(50, 2))
    X2 = np.random.normal(loc=[6, 6], scale=[1, 1], size=(50, 2))
    X3 = np.random.normal(loc=[10, -6], scale=[1, 1], size=(50, 2))
    X4 = np.random.normal(loc=[-6, 10], scale=[1, 1], size=(50, 2))
    X5 = np.random.normal(loc=[0, 0], scale=[1, 1], size=(50, 2))
    X_train = np.concatenate([X1, X2, X3, X4, X5])

    y1 = np.ones(50)
    y2 = 2 * np.ones(50)
    y3 = 3 * np.ones(50)
    y4 = 4 * np.ones(50)
    y5 = 5 * np.ones(50)
    y_train = np.concatenate([y1, y2, y3, y4, y5])

    svc = SVC(kernel='linear')

    draw(X_train, y_train, svc)    

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

    draw(X, Y, svc)


test_kernal()