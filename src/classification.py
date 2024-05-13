from data_preprocessing import load_data
from svm import SVC
from SVM import SupportVectorClassifier
from sklearn.svm import SVC as skSVC
import numpy as np
import time

def test_mnist():
    image_train, label_train, image_test, label_test = load_data()
    image_train = image_train.reshape(-1, 28*28)
    image_test = image_test.reshape(-1, 28*28)

    start = time.time()
    svc = SVC(C=1, tol=1e-4, max_passes=100, kernal='guassian',lang='python', threading=False, gamma=0.05)
    svc.ovo_fit(image_train, label_train)

    # svc = SupportVectorClassifier()
    # svc.fit(image_train, label_train)

    print('Start fitting')
    end = time.time()
    print('Finish fitting, Time: ', end-start)   
    y_pred = svc.predict(image_test)

    print('Accuracy: ', np.mean(y_pred == label_test))

test_mnist()