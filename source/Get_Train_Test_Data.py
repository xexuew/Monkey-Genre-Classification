import os
from os.path import join
import numpy as np
import pandas as pd
import h5py

from keras.preprocessing import image
from keras.applications import xception
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class GetTrainTestData(object):

    def __init__(self, config):
        
        self.PATH = config['PATH_CONFIGURATION']['DATASET_PATH']
        self.IMAGES_PATH = config['PATH_CONFIGURATION']['IMAGES_PATH']
        self.SIZE = int(config['IMAGE_FEATURES']['SIZE'])
        # Configuración de los datos
        self.SPLIT_SIZE = float(config['DATA_CONFIGURATION']['SPLIT_SIZE'])

    def getGenres(self):
        """
            Recoge los géneros del fichero de labels del dataset.
            # Return:
                genres: Array con las especies de los monos del dataset.
        """
        labels = pd.read_csv(join(self.PATH, 'monkey_labels.txt'))
        columns_new = ['Label', 'Latin_Name', 'Common_Name', 'Train_Images', 'Validation_Images']
        labels.columns = columns_new
        genres = [i.split() for i in labels.Common_Name.values]
        
        return genres
    

    def preprocessImage(self, path):
        """
            Preprocesamos cada imagen haciendo uso de xception
            # Return:
                img: Imagen ya preprocesada
        """
        img = image.load_img(path, target_size=(self.SIZE, self.SIZE))
        img = image.img_to_array(img)
        img = xception.preprocess_input(img.copy())
        return img
    

    def createDataset(self):

        # Obtenemos una lista de los directorios
        directorios = [nombre_directorio for nombre_directorio in os.listdir(self.IMAGES_PATH) \
                        if os.path.isdir(os.path.join(self.IMAGES_PATH, nombre_directorio))]
        directorios.sort()
        directorios.insert(0, directorios[0])

        # Escribimos el Dataset Preprocesado en formato h5py
        with h5py.File(self.PATH + 'dataset.hdf5', 'w') as hdf:            

            for root, subdirs, images in os.walk(self.IMAGES_PATH):
                # Ordenamos las carpetas por orden alfabético
                subdirs.sort() # Sort all subdirs

                try:
                    # Creamos un nuevo grupo con el nombre del directorio en el que estamos.
                    group_hdf = hdf.create_group(directorios[0]) 
                except Exception as e:
                    print("Error accured " + str(e))

                for img in tqdm(images):
                     # Descartamos otros ficheros .DS_store
                    if img.endswith('.jpg'):
                        try:
                            file_Path = os.path.join(root, img)
                            npy_image = self.preprocessImage(file_Path)
                            group_hdf.create_dataset(
                                img,
                                data=npy_image,
                                compression='gzip') # Incluimos el fichero numpy en el dataset.
                        except Exception as e:
                            print("Error accured" + str(e))
                directorios.n7(0) # Next directorio

    def getDataFromDataset(self, specie, dataset_file):
        """
            Recoge la imagen del archivo h5py
            # Arguments:
                specie: string
                    Nombre de la especie del que se desea obtener los arrays.
                dataset_file: h5py File
                    Dataset File que contiene los datos preprocesados.
            # Return:
                read_data: list(np.array)
                    Lista que contiene todos los arrays de la especie seleccionada.
        """
        # Lista que acumula los datos leidos del conjunto de datos.
        read_data = []

        print("Obteniendo.." + self.PATH + specie)

        # Leemos los datos
        for items in tqdm(dataset_file[specie]):
            read_data.append((dataset_file[specie][items][()]))

        return read_data

    def readDataset(self):

        dataset_file = h5py.File(self.PATH + 'dataset.hdf5', 'r')

        # Obtenemos las imagenes de cada especie
        arr_n0 = self.getDataFromDataset('n0', dataset_file)
        arr_n1 = self.getDataFromDataset('n1', dataset_file)
        arr_n2 = self.getDataFromDataset('n2', dataset_file)
        arr_n3 = self.getDataFromDataset('n3', dataset_file)
        arr_n4 = self.getDataFromDataset('n4', dataset_file)
        arr_n5 = self.getDataFromDataset('n5', dataset_file)
        arr_n6 = self.getDataFromDataset('n6', dataset_file)
        arr_n7 = self.getDataFromDataset('n7', dataset_file)
        arr_n8 = self.getDataFromDataset('n8', dataset_file)
        arr_n9 = self.getDataFromDataset('n9', dataset_file)

        # Los agrupamos
        full_data = np.vstack((arr_n0,\
                            arr_n1,\
                            arr_n2,\
                            arr_n3,\
                            arr_n4,\
                            arr_n5,\
                            arr_n6,\
                            arr_n7,\
                            arr_n8,\
                            arr_n9))

        # Establecemos las etiquetas que identifican el género musical
        labels = np.concatenate((np.zeros(len(arr_n0)),\
                                np.ones(len(arr_n1)),\
                                np.full(len(arr_n2), 2),\
                                np.full(len(arr_n3), 3),\
                                np.full(len(arr_n4), 4),\
                                np.full(len(arr_n5), 5),\
                                np.full(len(arr_n6), 6),\
                                np.full(len(arr_n7), 7),\
                                np.full(len(arr_n8), 8),\
                                np.full(len(arr_n9), 9)))

        # Con train_test_split() dividimos los datos.
        # Se puede cambiar el tamaño en el archivo config.
        print("test-size = " + str(self.SPLIT_SIZE) + " Cambiar valor en config.py")

        # Dividimos los datos, en función a SPLIT_SIZE (config)
        X_train, X_test, y_train, y_test = train_test_split(
            full_data,
            labels,
            test_size=self.SPLIT_SIZE,
            stratify=labels)

        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=0.5,
            stratify=y_test)

        print("X_train Tamaño: %s - X_test Tamaño: %s - X_val Tamaño: %s\
              - y_train Tamaño: %s - y_test Tamaño: %s - y_val Tamaño: %s " % \
             (X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape))

        return X_train, X_test, X_val, y_train, y_test, y_val
