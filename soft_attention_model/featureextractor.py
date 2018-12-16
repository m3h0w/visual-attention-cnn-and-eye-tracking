import keras

from keras.applications.resnet50 import ResNet50 as resnet
from keras.applications.resnet50 import preprocess_input as resnet_pp

from keras import backend as K
from keras.preprocessing import image

import skimage
import numpy as np

import gc

class FeatureExtractor:
    # Build a model by adding preprocessing before the pretrained CNN
    @staticmethod
    def get_feature_extraction_model(img_size):
        cnn_object, pp_function = FeatureExtractor._get_pretrained_model()
        model = keras.models.Sequential()
        cnn_model = cnn_object(weights='imagenet', include_top=False, pooling='max')
        model.add(keras.layers.Lambda(pp_function, name='preprocessing', input_shape=(img_size, img_size, 3)))
        model.add(cnn_model)
        return model

    # Unpack information from the models dictionary
    @staticmethod
    def _get_pretrained_model():
        cnn_object = resnet
        pp_function = resnet_pp
        return cnn_object, pp_function

    # generate patches from images
    @staticmethod
    def _get_patches(x, img_size, patch_width):
        patches = np.squeeze(np.asarray(skimage.util.view_as_windows(x, window_shape=(1,patch_width,patch_width,3), 
                                                                     step=(1,patch_width,patch_width,3)), dtype=np.int))
        len_ = x.shape[0]
#         del x
#         gc.collect()
        print(patches.shape)
        patches = patches.reshape(len_, int(img_size/patch_width)**2, patch_width, patch_width, 3)
        return patches

    # load an array of images given file names
    @staticmethod
    def _load_images(files, img_size):
        return np.asarray([image.img_to_array(image.load_img(file, target_size=(img_size, img_size))) for file in files], dtype=np.int)

    # get features for a list of file names (low level function)
    @staticmethod
    def get_features(files, model, img_size, patch_width):
        # Load images based on the size of the Lambda layer 
        # provided as the first layer before the pretrained CNN
        x = FeatureExtractor._load_images(files, img_size)
        patches = FeatureExtractor._get_patches(x, img_size, patch_width)
        patches_shape = patches.shape
        
        features = model.predict(patches.reshape(-1, patch_width, patch_width, 3), verbose=1)
        print(features.shape)
        
#         del x
#         del patches
#         gc.collect()
        
        return features.reshape(patches_shape[0], patches_shape[1], 2048)