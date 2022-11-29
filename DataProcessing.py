import numpy as np
import tensorflow as tf
from Nodes import Node
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator


def getNodes():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(32, 32),
        batch_size=20000,
        class_mode='binary')


    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            'data/test',
            target_size=(32, 32),
            batch_size=2500,
            class_mode='binary')

    (x_train, y_train) = train_generator.next()
    (x_test, y_test) = test_generator.next()

    nodes = []

    for i in range(100):
        start = 200*i
        end = 200*(i+1)
        n = Node(x_train[start:end], y_train[start:end], node_number=i)
        nodes.append(n)

    return nodes, x_test, y_test

def makeNodeMalicious(node_obj):
    x_train, y_train = node_obj.get_data()
    y_train_rand = np.random.randint(2, size=len(y_train))

    node_obj.set_data(x_train, y_train_rand)
