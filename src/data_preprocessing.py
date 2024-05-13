import numpy as np
import gzip
import os


mnist_path = 'data/'
save_path = 'data/mnist/'
def load_data():
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if os.path.exists(os.path.join(save_path, 'train_image.npy')):
        image_train = np.load(os.path.join(save_path, 'train_image.npy'))
        label_train = np.load(os.path.join(save_path, 'train_label.npy'))
        image_test = np.load(os.path.join(save_path, 'test_image.npy'))
        label_test = np.load(os.path.join(save_path, 'test_label.npy'))
        return image_train, label_train, image_test, label_test

    with gzip.open(os.path.join(mnist_path, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        image_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 255.0
    with gzip.open(os.path.join(mnist_path, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        label_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(mnist_path, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        image_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / 255.0
    with gzip.open(os.path.join(mnist_path, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        label_test = np.frombuffer(f.read(), np.uint8, offset=8)
    remain_train = 500
    remain_test = 100

    image_train_list, label_test_list = [], []
    for value in range(10):
        index = np.where(label_train == value)[0][:remain_train]
        image_train_list.append(image_train[index])
        label_test_list.append(label_train[index])
    image_train, label_train = np.concatenate(image_train_list), np.concatenate(label_test_list)
    index = np.random.permutation(len(label_train))
    image_train, label_train = image_train[index], label_train[index]

    image_test_list, label_test_list = [], []
    for value in range(10):
        index = np.where(label_test == value)[0][:remain_test]
        image_test_list.append(image_test[index])
        label_test_list.append(label_test[index])
    image_test, label_test = np.concatenate(image_test_list), np.concatenate(label_test_list)
    index = np.random.permutation(len(label_test))
    image_test, label_test = image_test[index], label_test[index]

    np.save(os.path.join(save_path, 'train_image.npy'), image_train)
    np.save(os.path.join(save_path, 'train_label.npy'), label_train)
    np.save(os.path.join(save_path, 'test_image.npy'), image_test)
    np.save(os.path.join(save_path, 'test_label.npy'), label_test)

    return image_train, label_train, image_test, label_test

