#!/usr/bin/env sh

# Model Training with Transfer Learning
/home/kb/caffe/build/tools/caffe train --solver=/home/kb/PlantVillage_Caffe/VGG16/solver.prototxt --weights /home/kb/PlantVillage_Caffe/VGG16/VGG_ILSVRC_16_layers.caffemodel 2>&1 | tee /home/kb/PlantVillage_Caffe/VGG16/pv.log

echo "done"
