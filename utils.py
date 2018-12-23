import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def unison_shuffled_copies(the_list):
    assert len(the_list) > 1
    for i in range(len(the_list) - 1):
        assert len(the_list[i]) == len(the_list[i+1])
    p = np.random.permutation(len(the_list[0]))
    return [el[p] for el in the_list]

def normalize(a):
    return (a - np.min(a))/np.ptp(a)

def _get_initial_lstm(features, H, D=2048):
    with tf.variable_scope('initial_lstm'):
        features_mean = tf.reduce_mean(features, 1)

        w_h = tf.get_variable('w_h', [D, H])
        b_h = tf.get_variable('b_h', [H])
        h = tf.nn.tanh(tf.matmul(tf.ones_like(features_mean), w_h) + b_h)

        w_c = tf.get_variable('w_c', [D, H])
        b_c = tf.get_variable('b_c', [H])
        c = tf.nn.tanh(tf.matmul(tf.ones_like(features_mean), w_c) + b_c)
        return [c, h]

# filters out of a list all paths that are not in a list of filenames in ids_list
def filter_by_ids(files, ids_list):       
    return [file for file in files if filename_from_path(file) in ids_list]

def filename_from_path(path):
    return path.split('/')[-1]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt.gcf()