from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.utils import normalize


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os

import numpy as np
from scipy import misc


def setKey(dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = [value]
        elif type(dictionary[key]) == list:
            dictionary[key].append(value)
        else:
            dictionary[key] = [dictionary[key], value]

class DATA_SET():
    def __init__(self, directory, grayScale=True, labels=True, img_rows = 64,img_cols = 64):
        self.img_rows = img_rows
        self.img_cols = img_cols

        self.labels = labels
        if grayScale == True:
            self.channels = 1 # 1 channel greyscale
        else:
            self.channels = 3 # 3 channels RGB
	
        self.num_classes = 0

        self.pathes = {}
        self.signs = []
        self.dataSetLen = 0
        self.minNumberOfImages = 100

        if(self.labels == True):
            self.Y_train_list = []

        forbiddenString = "G1,G2,G3,G4,G5,I1,IJ4b,IP1a,IP1b,IP4a,IP9,X1,X2,X3,Z4a,Z4b,Z4c,Z4e,Z5d,Z9"
        self.listOfForbidden = forbiddenString.split(",")
        print (self.listOfForbidden)
        # Not really helpful sings:
        '''
        G1,G2,G3,G4,G5
        I1, IJ4b
        IP1a, IP1b, IP4a, IP9,
        IS - the whole- information signs with names of cities etc... not necessary - it would bring too much variability
        X1, X2, X3
        All Zs except Z3 
        '''
        for root, folders, files in os.walk(directory):

            if len(files) > self.minNumberOfImages and root != '.':
                sign = os.path.split(root)[-1]  
                # creating list of signs names

                if (sign != "dataSet_mini") or (len(sign) < 1):
                    # Preventing strange classes to be part of my data-set
                    if (sign not in self.listOfForbidden) and ("IS" not in sign):                 

                        self.signs.append(sign)
                        impath = os.path.join("dataSet_mini", sign)
                        self.num_classes += 1

                        for file in os.listdir(impath):
                            if file.endswith(".jpg") or file.endswith(".png"):
                                # adding into dictionary image pathes
                                img_rgb = misc.imread(os.path.join(impath, file), mode='RGB')
                                w,h,d = img_rgb.shape
                                if 47 < w or 47 < h:
                                    setKey(self.pathes, sign, str(os.path.join(impath, file)))
                                    self.dataSetLen += 1
                                    if(self.labels == True):
                                        self.Y_train_list.append(self.num_classes)
                                else:
                                    print ("Too small resolution: {}".format(file))
                                    os.remove(os.path.join(impath, file))

        print("Length of dataset is: {}".format(self.dataSetLen))
        print("Number of classes: {}".format(self.num_classes))

        self.training_shape = (self.dataSetLen, self.img_rows, self.img_cols, self.channels)
        print("Training shape: {}".format(self.training_shape))
        
        self.X_train_pos = np.zeros(shape=self.training_shape, dtype=np.uint8)
        self.X_train_neg = np.zeros(shape=self.training_shape, dtype=np.uint8)

        self.X_test_pos = np.zeros(shape=self.training_shape, dtype=np.uint8)
        self.X_test_neg = np.zeros(shape=self.training_shape, dtype=np.uint8)

        self.Y_train = np.zeros(shape=self.dataSetLen, dtype=np.uint8)
        self.Y_test = np.zeros(shape=self.dataSetLen, dtype=np.uint8)
            
    #np.zeros(shape=(numberOfSatelites, 3), dtype=float)


    def loadImages(self):
        allIndexPos = 0
        allIndexNeg = 0
        # defining ration between positive and negative samples
        for sign, empty in self.pathes.items():
            for imgPath in self.pathes[sign]:
                #print (imgPath)
                if self.labels == True:
                    self.Y_train[allIndexPos] = self.Y_train_list[allIndexPos]

                if self.channels == 1:
                    img_grey = misc.imread(imgPath, mode='L')
                    img_grey = misc.imresize(img_grey, size=(self.img_rows,self.img_cols,self.channels), interp='bilinear', mode=None)
                else:
                    img_rgb = misc.imread(imgPath, mode='RGB')
                    img_rgb = misc.imresize(img_rgb, size=(self.img_rows,self.img_cols,self.channels), interp='bilinear', mode=None)
                
                if self.channels == 1:
                    self.X_train_pos[allIndexPos,:,:,0] = ((img_grey.astype(np.float32)/ 255) - 0.5) * 2

                else:
                    self.X_train_pos[allIndexPos,:,:,:] = (img_rgb.astype(np.float32) - 127.5) / 127.5 

                allIndexPos += 1

        return allIndexPos, self.num_classes
