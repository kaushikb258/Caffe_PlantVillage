#!/usr/bin/env sh

PVDIR="/home/kb/PlantVillage_Caffe"
CAFFE_ROOT="/home/kb/caffe"

python $PVDIR/codes/create_train_val.py
python $PVDIR/codes/create_lmdb.py 

# compute mean
$CAFFE_ROOT/build/tools/compute_image_mean $PVDIR/lmdb/train_lmdb $PVDIR/lmdb/mean.binaryproto

# create a png graphic of the convnet
python $CAFFE_ROOT/python/draw_net.py $PVDIR/VGG16/train_val.prototxt $PVDIR/VGG16/VGG16.png

echo "done"
