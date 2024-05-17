import numpy as np
import gzip
import os
import pickle
from skimage.feature import hog as skhog
import cv2
from tqdm import tqdm


def load_mnist_data():
    mnist_path = 'data/'
    save_path = 'data/mnist/'
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

def load_cifar10_hog():
    path = 'data/cifar-10-batches-py/'
    save_path = 'data/cifar-10-hog/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, 'train_hog.npy')):
        train_hog = np.load(os.path.join(save_path, 'train_hog.npy'))
        train_label = np.load(os.path.join(save_path, 'train_label.npy'))
        test_hog = np.load(os.path.join(save_path, 'test_hog.npy'))
        test_label = np.load(os.path.join(save_path, 'test_label.npy'))
        return train_hog, train_label, test_hog, test_label
    
    train_img = []
    train_label = []
    for i in range(1, 6):
        with open(os.path.join(path, f'data_batch_{i}'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        train_img.append(np.array(data[b'data'], dtype=np.float32).reshape(-1,3,32,32) / 255.)
        train_label.append(np.array(data[b'labels'], dtype=np.int32))
    train_img = np.concatenate(train_img)
    train_label = np.concatenate(train_label)

    with open(os.path.join(path, 'test_batch'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    test_img = np.array(data[b'data'], dtype=np.float32).reshape(-1,3,32,32) / 255.
    test_label = np.array(data[b'labels'], dtype=np.int32)

    train_img_select = []
    train_label_select = []
    test_img_select = []
    test_label_select = []
    for i in range(10):
        index = np.where(train_label == i)[0][:500]
        train_img_select.append(train_img[index])
        train_label_select.append(train_label[index])
        index = np.where(test_label == i)[0][:100]
        test_img_select.append(test_img[index])
        test_label_select.append(test_label[index])
    train_img = np.concatenate(train_img_select)
    train_label = np.concatenate(train_label_select)
    test_img = np.concatenate(test_img_select)
    test_label = np.concatenate(test_label_select)
    train_hog = train_img
    test_hog = test_img
    index = np.random.permutation(len(train_label))
    train_hog, train_label = train_hog[index], train_label[index]
    index = np.random.permutation(len(test_label))
    test_hog, test_label = test_hog[index], test_label[index]
    train_hog, test_hog = BatchHOG(train_hog, partition=16), BatchHOG(test_hog, partition=16)
    np.save(os.path.join(save_path, 'train_hog.npy'), train_hog)
    np.save(os.path.join(save_path, 'train_label.npy'), train_label)
    np.save(os.path.join(save_path, 'test_hog.npy'), test_hog)
    np.save(os.path.join(save_path, 'test_label.npy'), test_label)
    return train_hog, train_label, test_hog, test_label

def HOG(image, block=4, partition=8):
    image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    height, width = image.shape
    gradient = np.zeros((2, height, width), dtype=np.float32)
    for i in range(1, height-1):
        for j in range(1, width-1):
            delta_x = image[i, j-1] - image[i, j+1]
            delta_y = image[i+1, j] - image[i-1, j]
            gradient[0, i, j] = np.sqrt(delta_x ** 2 + delta_y ** 2)
            gradient[1, i, j] = np.degrees(np.arctan2(delta_y, delta_x))
            if gradient[1, i, j] < 0:
                gradient[1, i, j] += 180
    unit = 360 / partition
    vertical, horizontal = height // block, width // block
    feature = np.zeros((vertical, horizontal, partition), dtype=np.float32)
    for i in range(vertical):
        for j in range(horizontal):
            for k in range(block):
                for l in range(block):
                    rho = gradient[0, i*block+k, j*block+l]
                    theta = gradient[1, i*block+k, j*block+l]
                    index = int(theta // unit)
                    feature[i, j, index] += rho
            feature[i, j] /= np.linalg.norm(feature[i, j]) + 1e-6
    return feature.reshape(-1)


def BatchHOG(images, block=4, partition=8):
    feature_list = []
    for image in tqdm(images):
        feature_list.append(HOG(image, block, partition))
    return np.array(feature_list)

