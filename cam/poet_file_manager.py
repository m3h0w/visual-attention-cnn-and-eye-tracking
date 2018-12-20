import os, sys
sys.path.insert(0,'..') # add parent directory to path

import glob
import numpy as np
from keras.preprocessing import image as kimage
import skimage
from keras.utils.np_utils import to_categorical

import utils

def save_poet_to_folders(img_size, mode, ids_list, input_channels=3, main_path = r'/home/mikey/Data/POETdataset/PascalImages/'):
    def load_images(files, grayscale=False):
        if grayscale:
            return (np.array([kimage.img_to_array(kimage.load_img(file, target_size=img_size, color_mode='grayscale'), dtype=np.uint8) 
                             for file in files]))
        else:
            return (np.array([kimage.img_to_array(kimage.load_img(file, target_size=img_size), dtype=np.uint8) 
                             for file in files]))
    
    grayscale = True
    if input_channels == 3:
        grayscale = False
    
    print("Loading POET dataset...")
    
    classes = ['aeroplane', 'boat', 'dog', 'bicycle', 'cat', 'cow', 'diningtable', 'horse', 'motorbike','sofa']
    classes = [word + '*' for word in classes]
    files_list = [glob.glob(main_path + class_) for class_ in classes]
    
    for files in files_list:
        assert len(files) > 0
    
    files_dict = {class_name: class_files for class_name, class_files in zip(classes, files_list)}
    for class_number, (class_name, files) in enumerate(files_dict.items()):
        print(len(files))
        files = utils.filter_by_ids(files, ids_list)
        print(len(files))
        one_class_images = load_images(files)

        new_dir = './' + mode + '/' + str(class_number)
        print(new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        
        for img, name in zip(one_class_images, [file.split('/')[-1] for file in files]):
#             print(np.max(img), np.min(img))
            skimage.io.imsave(new_dir + '/' + name, img)

def load_poet(img_size, keep_ids, input_channels=3, main_path = r'/home/mikey/Data/POETdataset/PascalImages/'):
    def load_images(files, grayscale=False):
        if grayscale:
            return np.array([kimage.img_to_array(kimage.load_img(file, target_size=img_size, color_mode='grayscale')) 
                             for file in files])
        else:
            return np.array([kimage.img_to_array(kimage.load_img(file, target_size=img_size)) 
                             for file in files])
    
    grayscale = True
    if input_channels == 3:
        grayscale = False
    
    print("Loading POET dataset...")
    
    classes = ['aeroplane', 'boat', 'dog', 'bicycle', 'cat', 'cow', 'diningtable', 'horse', 'motorbike','sofa']
    classes = [word + '*' for word in classes]
    files_list = [glob.glob(main_path + class_) for class_ in classes]
    
    for files in files_list:
        assert len(files) > 0
    
    x = []
    y = []
    names = []
    class_map = {}
    files_dict = {class_name: class_files for class_name, class_files in zip(classes, files_list)}
    for class_number, (class_name, files) in enumerate(files_dict.items()):
        files = utils.filter_by_ids(files, keep_ids)
        x.append(load_images(files))
        y.append([class_number] * len(files))
        print(class_name, class_number)
        class_map[class_name.replace('*', '')] = class_number
        names.append([utils.filename_from_path(file) for file in files])
        
    X_images = np.concatenate(x)
    y = np.concatenate(y)
    print('X.shape:', X_images.shape, 'y.shape:', y.shape)

    print("Loaded POET dataset.")
    return X_images, to_categorical(y), class_map, np.concatenate(names)