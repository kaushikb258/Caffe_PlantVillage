#!/usr/bin/env sh

# Model Training with Transfer Learning
/home/kb/caffe/build/tools/caffe train --solver=/home/kb/PlantVillage_Caffe/AlexNet/solver.prototxt --weights /home/kb/PlantVillage_Caffe/AlexNet/bvlc_alexnet.caffemodel 2>&1 | tee /home/kb/PlantVillage_Caffe/AlexNet/pv.log

echo "done"
