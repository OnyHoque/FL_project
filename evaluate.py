import joblib
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

model = joblib.load("Generic.model")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(32, 32),
        batch_size=2500,
        class_mode='binary')

(x_test, y_test) = test_generator.next()

result = model.evaluate(x_test, y_test)
print(result)

y_pred = model.predict(x_test)

y1 = y_test
y2 = y_pred.argmax(1)
print('Precision: %.3f' % precision_score(y1, y2, average='micro'))
print('Recall: %.3f' % recall_score(y1, y2, average='micro'))
print('F1: %.3f' % f1_score(y1, y2, average='micro'))
print('Accuracy: %.3f' % accuracy_score(y1, y2))