import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def evaluate(history, prediction, original):
    c = confusion_matrix([np.argmax(y) for y in original], [np.argmax(y) for y in prediction])
    print(c / c.astype(np.float).sum(axis=1))

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
