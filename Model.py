# from tensorflow.keras.applications import resnet50, densenet, efficientnet, inception_v3, mobilenet_v3, vgg16
import tensorflow as tf
from keras import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,BatchNormalization,Dropout


def getModel(model_name):
    model = Sequential()
    keras_model = None

    if model_name == "ResNet50":
        keras_model = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', input_shape = (32,32,3))
        
    if model_name == "DenseNet121":
        keras_model = tf.keras.applications.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (32,32,3))

    if model_name == "MobileNet":
        keras_model = tf.keras.applications.MobileNet(include_top = False, weights = 'imagenet', input_shape = (32,32,3))

    if model_name == "VGG16":
        keras_model = tf.keras.applications.VGG16(include_top = False, weights = 'imagenet', input_shape = (32,32,3))

    if model_name == "InceptionV3":
        keras_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet', input_shape = (32,32,3))

    if model_name == "EfficientNetB7":
        keras_model = tf.keras.applications.EfficientNetB7(include_top = False, weights = 'imagenet', input_shape = (32,32,3))
    
    if model_name == "Generic":
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(1,activation='sigmoid'))
        sgd = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=sgd, loss='binary_crossentropy',  metrics=['accuracy'])
        return model

    out=keras_model.layers[-1].output
    out = Flatten()(out)
    output = Dense(1, activation='sigmoid')(out)
    model = tf.keras.models.Model(inputs=keras_model.input, outputs=output)
    sgd = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy',  metrics=['accuracy'])

    return model