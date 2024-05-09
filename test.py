import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from svm import SVC
from sklearn.svm import SVC as skSVC

# Create a SVC instance
# svc = skSVC(kernel='linear')
svc = SVC()


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
X1 = np.random.normal(loc=[2, 2], scale=[1, 1], size=(50, 2))
X2 = np.random.normal(loc=[6, 6], scale=[1, 1], size=(50, 2))
X_train = np.concatenate([X1, X2])

y1 = np.ones(50)
y2 = -np.ones(50)
y_train = np.concatenate([y1, y2])

# Fit the model
svc.fit(X_train, y_train)

# Create a grid to evaluate model
xx = np.linspace(0, np.max(X_train), 100)
# w, b = svc.coef_[0], svc.intercept_
w, b = svc.param()
print(w, b)

# Calculate decision boundary (z = wx + b)
yy = (-w[0]/w[1]) * xx - b/w[1]

# Plot decision boundary
plt.plot(xx, yy)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.savefig('2d_decision_boundary.png')