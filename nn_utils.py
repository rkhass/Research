import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

def get_pr_auc_score(y_true, y_pred):
    value = np.round(average_precision_score(y_true, y_pred), 4)
    print('PR AUC score:',  value)
    
    return value

def plot_history(history):
    acc = history.history['auroc']
    val_acc = history.history['val_auroc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def aucroc(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.double)