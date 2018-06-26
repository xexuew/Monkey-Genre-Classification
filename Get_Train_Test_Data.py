import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.applications import xception

from tqdm import tqdm
import os

from os.path import join

class GetTrainTestData(object):

    def __init__(self, config):
        
        self.PATH = config['PATH_CONFIGURATION']['DATASET_PATH']
        self.TRAIN_PATH = config['PATH_CONFIGURATION']['TRAIN_IMAGES_PATH']
        self.VAL_PATH = config['PATH_CONFIGURATION']['VAL_IMAGES_PATH']

        self.SIZE = int(config['IMAGE_FEATURES']['SIZE'])

    #
    # Description:
    # Input:
    # Output:
    def get_genres(self):

        labels = pd.read_csv(join(self.PATH, 'monkey_labels.txt'))
        columns_new = ['Label', 'Latin_Name', 'Common_Name', 'Train_Images', 'Validation_Images']
        labels.columns = columns_new
        genres = [i.split() for i in labels.Common_Name.values]
        
        return genres
    
    #
    # Description:
    # Input:
    # Output:
    def preprocess_img(self, path):
        img = image.load_img(path, target_size=(self.SIZE, self.SIZE))
        img = image.img_to_array(img)
        img = xception.preprocess_input(img.copy())
        return img
    
    #
    # Description:
    # Input:
    # Output:
    def prepossessingImages(self, option):
        data_arr = []
        labels_arr = []

        if option == 'train':
            path = self.TRAIN_PATH
        elif option == 'val':
            path = self.VAL_PATH
        else:
            print("Error to get Train or Val")
            
            
        for root, subdirs, images in os.walk(path):
            subdirs.sort() # Sort all subdirs
            labels_arr.append(len(images))

            for img in tqdm(images):
                if img.endswith('.jpg'):
                    try:
                        file_Path = os.path.join(root, img)
                        npy_image = self.preprocess_img(file_Path)
                        data_arr.append(npy_image)

                    except Exception as e:
                        print("Error accured" + str(e))
                else:
                    labels_arr[-1] = labels_arr[-1]-1 #File error
            
        labels = np.concatenate((np.zeros(labels_arr[1]),\
                            np.ones(labels_arr[2]),\
                            np.full(labels_arr[3], 2),\
                            np.full(labels_arr[4], 3),\
                            np.full(labels_arr[5], 4),\
                            np.full(labels_arr[6], 5),\
                            np.full(labels_arr[7], 6),\
                            np.full(labels_arr[8], 7),\
                            np.full(labels_arr[9], 8),\
                            np.full(labels_arr[10], 9)))
        
        return np.array(data_arr), labels