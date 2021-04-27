import tensorflow as tf
import numpy as np
import math


def load_MNIST() -> dict:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    d = {}
    d['name'] = 'mnist'
    d['x_train'] = np.expand_dims(x_train, -1)
    d['y_train'] = y_train
    d['x_test'] = np.expand_dims(x_test, -1)
    d['y_test'] = y_test
    d['num_class'] = int(max(y_train) + 1)
    d['class_labels'] = np.array(
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    return d


def load_Fashion_MNIST() -> dict:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    d = {}
    d['name'] = 'fashion_mnist'
    d['x_train'] = np.expand_dims(x_train, -1)
    d['y_train'] = y_train
    d['x_test'] = np.expand_dims(x_test, -1)
    d['y_test'] = y_test
    d['num_class'] = int(max(y_train) + 1)
    d['class_labels'] = np.array([
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ])
    return d


def load_CIFAR10() -> dict:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    d = {}
    d['name'] = 'cifar10'
    d['x_train'] = x_train
    d['y_train'] = y_train.squeeze()
    d['x_test'] = x_test
    d['y_test'] = y_test.squeeze()
    d['num_class'] = int(max(y_train) + 1)
    d['class_labels'] = np.array([
        "airplain", "automobile", "bird", "cat", "deer", "dog", "frog",
        "horse", "ship", "truck"
    ])
    return d


def load_CIFAR100() -> dict:
    (x_train, y_train), (x_test,
                         y_test) = tf.keras.datasets.cifar100.load_data()
    d = {}
    d['name'] = 'cifar100'
    d['x_train'] = x_train
    d['y_train'] = y_train.squeeze()
    d['x_test'] = x_test
    d['y_test'] = y_test.squeeze()
    d['num_class'] = int(max(y_train) + 1)
    d['class_labels'] = np.array([
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
        'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
        'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
        'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
        'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
        'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
        'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
        'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ])
    return d


def load_Dataset(dataset: str, x_normalization=True, y_one_hot=True, verbose=True) -> dict:
    if dataset == 'mnist':
        d = load_MNIST()
    elif dataset == 'fashion_mnist':
        d = load_Fashion_MNIST()
    elif dataset == 'cifar10':
        d = load_CIFAR10()
    elif dataset == 'cifar100':
        d = load_CIFAR100()
    else:
        return {}
    num_class = d['num_class']
    if x_normalization:
        d['x_train'] = d['x_train'].astype(np.float32) / 255.0
        d['x_test'] = d['x_test'].astype(np.float32) / 255.0
    if y_one_hot:
        d['y_train'] = tf.keras.utils.to_categorical(d['y_train'], num_class)
        d['y_test'] = tf.keras.utils.to_categorical(d['y_test'], num_class)
    d['input_shape'] = d['x_train'].shape[1:]  # (H, W, C)
    if verbose:
        x_train, x_test = d['x_train'], d['x_test']
        y_train, y_test = d['y_train'], d['y_test']
        print(f'dataset={dataset}, num_class={num_class}')
        pstr = "[{}] images: shape={}, dtype={}; labels: shape={}, dtype={}"
        print(pstr.format('Train', x_train.shape,
                          x_train.dtype, y_train.shape, y_train.dtype))
        print(pstr.format('Test', x_train.shape,
                          x_train.dtype, y_train.shape,  y_train.dtype))
    return d
