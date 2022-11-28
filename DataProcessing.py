import numpy as np
import tensorflow as tf
from Nodes import Node
import keras.utils.np_utils as kutils


def getNodes():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    np.random.shuffle(x_train)
    np.random.shuffle(y_train)
    np.random.shuffle(x_test)
    np.random.shuffle(y_test)

    x_train = x_train/255.0
    x_test = x_test/255.0
    y_train = kutils.to_categorical(y_train, 10)
    y_test = kutils.to_categorical(y_test, 10)

    nodes = []

    for i in range(100):
        start = 500*i
        end = 500*(i+1)
        n = Node(x_train[start:end], y_train[start:end], node_number=i)
        nodes.append(n)

    return nodes, x_test, y_test

def makeNodeMalicious(node_obj):
    x_train, y_train = node_obj.get_data()
    y_train_rand = np.random.randint(10, size=len(y_train))
    y_train_rand = kutils.to_categorical(y_train_rand, 10)
    node_obj.set_data(x_train, y_train_rand)
