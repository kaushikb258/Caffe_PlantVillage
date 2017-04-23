'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH=227
IMAGE_HEIGHT=227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())


train_lmdb = '/home/kb/PlantVillage_Caffe/lmdb/train_lmdb'
val_lmdb = '/home/kb/PlantVillage_Caffe/lmdb/val_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + val_lmdb)


DATADIR = "/home/kb/PlantVillage_Caffe/data" 
TXTFILE = "/home/kb/PlantVillage_Caffe/lmdb"



print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
counter = 0
with in_db.begin(write=True) as in_txn:
    f = open(TXTFILE + "/train.txt", "r")
    for line in f:
     ll = line.split(" ")
     img_path = ll[0]
     img_class = int(ll[1])
     if (img_class < 0 or img_class > 37):
       print "error!"   
     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
     img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
     datum = make_datum(img, img_class)
     in_txn.put('{:0>5d}'.format(counter), datum.SerializeToString())
     counter += 1
in_db.close()
print "total number of train images = ", counter


print 'Creating val_lmdb'

in_db = lmdb.open(val_lmdb, map_size=int(1e12))
counter = 0
with in_db.begin(write=True) as in_txn:
    f = open(TXTFILE + "/val.txt", "r")
    for line in f:
     ll = line.split(" ")
     img_path = ll[0]
     img_class = int(ll[1])   
     if (img_class < 0 or img_class > 37):
       print "error!" 
     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
     img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
     datum = make_datum(img, img_class)
     in_txn.put('{:0>5d}'.format(counter), datum.SerializeToString())
     counter += 1
in_db.close()
print "total number of val images = ", counter




print '\nFinished processing all images'
