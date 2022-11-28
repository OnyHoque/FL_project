# from tensorflow.keras.applications import resnet50, densenet, efficientnet, inception_v3, mobilenet_v3, vgg16
import tensorflow as tf
from keras import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


def getModel(model_name):
    model = Sequential()
    # feature_extractor = None

    # if model_name == "ResNet50":
    #     feature_extractor = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', input_shape = (32,32,3), classes = 10)
        
    # if model_name == "DenseNet121":
    #     feature_extractor = tf.keras.applications.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (32,32,3), classes = 10)

    # if model_name == "MobileNet":
    #     feature_extractor = tf.keras.applications.MobileNet(include_top = False, weights = 'imagenet', input_shape = (32,32,3), classes = 10)

    # if model_name == "VGG16":
    #     feature_extractor = tf.keras.applications.VGG16(include_top = False, weights = 'imagenet', input_shape = (32,32,3), classes = 10)

    # if model_name == "InceptionV3":
    #     feature_extractor = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet', input_shape = (32,32,3), classes = 10)

    # if model_name == "EfficientNetB7":
    #     feature_extractor = tf.keras.applications.EfficientNetB7(include_top = False, weights = 'imagenet', input_shape = (32,32,3), classes = 10)
    

    # model.add(feature_extractor)
    # model.add(Flatten())
    # model.add(Dense(10, activation='softmax'))
    # optimizer = SGD(learning_rate=0.001, momentum=0.9, decay=0.0, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.add(Conv2D(32,3,3,input_shape=(32,32,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(output_dim=128,activation='relu'))
    model.add(Dense(output_dim=1,activation='sigmoid'))


    return model