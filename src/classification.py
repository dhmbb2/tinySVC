from data_preprocessing import load_mnist_data, load_cifar10_hog
from svm import SVC
from sklearn.svm import SVC as skSVC
import numpy as np
import time

def test_mnist(C=2**-5, kernel='linear', gamma_poly=1, degree=2, gamma=1, heu=True, strategy='ovo'):
    image_train, label_train, image_test, label_test = load_mnist_data()
    image_train = image_train.reshape(-1, 28*28)
    image_test = image_test.reshape(-1, 28*28)

    start = time.time()
    svc = SVC(C=C, tol=1e-4, max_passes=100, kernel=kernel, degree=degree, gamma_poly=gamma_poly, gamma=gamma, heu=heu ,strategy=strategy)
    svc.fit(image_train, label_train)

    print('Start fitting')
    end = time.time()
    print('Finish fitting, Time: ', end-start)   
    y_pred = svc.predict(image_test)
    acc = np.mean(y_pred == label_test)

    train_pred = svc.predict(image_train)
    train_acc = np.mean(train_pred == label_train)
    print('Train acc: ', train_acc)
    print('Test acc: ', acc)

    return acc, train_acc

def test_hog(C=1, kernel='linear', gamma_poly=1, degree=2, gamma=1, heu=True, strategy='ovo'):
    image_train, label_train, image_test, label_test = load_cifar10_hog()
    start = time.time()
    svc = SVC(C=C, tol=1e-4, max_passes=100, kernel=kernel, degree=degree, gamma_poly=gamma_poly, gamma=gamma, heu=heu ,strategy=strategy)
    svc.fit(image_train, label_train)

    print('Start fitting')
    end = time.time()
    print('Finish fitting, Time: ', end-start)
    y_pred = svc.predict(image_test)
    acc = np.mean(y_pred == label_test)

    train_pred = svc.predict(image_train)
    train_acc = np.mean(train_pred == label_train)
    print('Train acc: ', train_acc)
    print('Test acc: ', acc)

    return acc, train_acc
