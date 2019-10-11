#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import matplotlib.pyplot as plt

from FID.fid_files import get_fid, get_inception_activations_original
from FID.fid import create_inception_graph, calculate_activation_statistics, calculate_frechet_distance, check_or_download_inception


def calculate_FID(real_image_path, generated_image_paths):

    paths = []
    for image_path in generated_image_paths:
        data = []

        image_list_jpg = glob.glob(os.path.join(real_image_path, '*.jpg'))

        image_list_png = glob.glob(os.path.join(real_image_path, '*.png'))

        image_list = image_list_jpg + image_list_png

        real_images = np.array([imresize(imread(str(fn)),size=(64,64,3), interp='bilinear', mode=None).astype(np.float32) for fn in image_list])
        real_images = np.moveaxis(real_images, -1, 1)
        real_images = np.moveaxis(real_images, -1, 2)
        real_images_crop = real_images[:1000,:,:,:]


        ### --------------- Calculate FID --------------
        # open file to store values
        lastString_file = image_path.split("/")[-1]
        file = open(lastString_file + "_FID.txt","w")

        data = []
        act = get_inception_activations_original(real_images_crop)
        for networks in image_path:
            # loads all images into memory (this might require a lot of RAM!
            print(networks)
            image_list_jpg = glob.glob(os.path.join(networks, '*.jpg'))

            image_list_png = glob.glob(os.path.join(networks, '*.png'))

            image_list = image_list_jpg + image_list_png
            print (image_list)
            fake_images = np.array([imresize(imread(str(fn)),size=(64,64,3), interp='bilinear', mode=None).astype(np.float32) for fn in image_list])
            fake_images = np.moveaxis(fake_images, -1, 1)
            fake_images = np.moveaxis(fake_images, -1, 2)
            fake_images_crop = real_images[:1000,:,:,:]

            fid = get_fid(act, real_images_crop)

            lastString = networks.split("/")[-1]

            print("FID for {} is: {}".format(lastString, fid))
            data.append(fid)
            file.write(str(fid))


        t = np.arange(0, 10000, 100)
        dataArray = np.asarray(data) 
        color = 'tab:red'
        linestyle = '-'
        plt.title('FID vs. Epochs')
        plt.xlabel('epochs')
        plt.ylabel('FID', color=color)
        plt.plot(t, dataArray, color=color, linestyle=linestyle)
        plt.savefig('FID_vs_Epochs_' + lastString_file + '.eps', format='eps')
        plt.savefig('FID_vs_Epochs_' + lastString_file + '.png')
        file.close()
        #plt.show()

