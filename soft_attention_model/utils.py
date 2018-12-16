import pickle
import numpy as np
import tensorflow as tf

class Utils:
    @staticmethod
    def save_object(obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load_object(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
        
    @staticmethod
    def unison_shuffled_copies(the_list):
        assert len(the_list) > 1
        for i in range(len(the_list) - 1):
            assert len(the_list[i]) == len(the_list[i+1])
        p = np.random.permutation(len(the_list[0]))
        return [el[p] for el in the_list]

    @staticmethod
    def normalize(a):
        return (a - np.min(a))/np.ptp(a)
    
    @staticmethod
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