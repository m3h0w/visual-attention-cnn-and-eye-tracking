import os, sys
sys.path.insert(0,'..') # add parent directory to path
import utils

class Config:
    main_path = os.path.abspath("/home/mikey/Data/POETdataset/PascalImages/")
    class_names = ['dog', 'aeroplane', 'boat', 'bicycle', 'cat', 'cow', 'diningtable', 'horse', 'motorbike','sofa']
    IMG_SIZE = 60*4*2
    PATCH_WIDTH = 60
    SMALLER_IMG_SIZE = 150
    T = int(IMG_SIZE/PATCH_WIDTH)**2
    new_dir = 'soft_attention_features_' + str(PATCH_WIDTH)
    new_dir_img = 'soft_attention_images_' + str(PATCH_WIDTH)
    name_to_class_dict = {class_name: i for i, class_name in enumerate(class_names)}
    train_ids, test_ids = utils.load_object('../train_ids.pkl'), utils.load_object('../test_ids.pkl')