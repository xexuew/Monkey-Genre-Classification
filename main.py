import configparser
import numpy as np

from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

from Get_Train_Test_Data import GetTrainTestData
from CNN_Model import CNNModel


config = configparser.ConfigParser()
config.read('config.ini')

X_train, y_train = GetTrainTestData(config).prepossessingImages('train')
X_test, y_test = GetTrainTestData(config).prepossessingImages('val')

X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size = 0.5,
            random_state = 0,
            stratify = y_test)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)
print("X_val: ", X_val.shape)
print("y_val: ", y_val.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)

model = CNNModel(config, X_train).build_model()

sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov=True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model.fit(
        X_train,
        y_train,
        batch_size = int(config['CNN_CONFIGURATION']['BATCH_SIZE']),
        epochs = int(config['CNN_CONFIGURATION']['EPOCHS']),
        verbose = 1,
        validation_data = (X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights('/output/weights.hdf5')

y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)

print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))