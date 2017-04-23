import glob
import os
import random
import shutil
import numpy as np

TRAIN_PERCENTAGE = 70

TRAIN_SET = []
VAL_SET = []

pvdir = "/home/kb/PlantVillage_Caffe"




try:
  shutil.rmtree(pvdir + "/data/train_images") 
except:
  pass
os.mkdir(pvdir + "/data/train_images")

try:
  shutil.rmtree(pvdir + "/data/val_images") 
except:
  pass
os.mkdir(pvdir + "/data/val_images")





nclasses = 38
netrain = np.zeros(nclasses,np.int)
neval = np.zeros(nclasses,np.int)


#Distribute the files into Training and Validation sets
filename = pvdir + "/data/crowdai/*/*"
for _image in glob.glob(filename):
  className = _image.split("/")[-2]

# Some fileNames contain spaces, which creates some incompatibility with a preprocessing script shipped with caffe
# Hence we replace all spaces in the filename with _
  newFileName = _image.split("/")[-1]
  newFileName = newFileName.replace(" ", "_")
  newFileName = newFileName.split("___")[1]

  if random.randint(0,100) < TRAIN_PERCENTAGE:
    newFilePath = pvdir + "/data/train_images/" + newFileName 
    shutil.copy(_image, newFilePath)
    TRAIN_SET.append((newFilePath, className.split("_")[-1]))
    i = int(className.split("_")[-1])
    netrain[i] += 1 
  else:
    newFilePath = pvdir + "/data/val_images/" + newFileName 
    shutil.copy(_image, newFilePath)
    VAL_SET.append((newFilePath, className.split("_")[-1]))
    i = int(className.split("_")[-1])
    neval[i] += 1


np.random.shuffle(TRAIN_SET)
np.random.shuffle(VAL_SET)


print "netrain = ", netrain
print "neval = ", neval


#Write the distribution into separate text files
try:
  os.mkdir(pvdir + "/lmdb")
except:
  pass

f = open(pvdir + "/lmdb/train.txt", "w")
for _entry in TRAIN_SET:
 f.write(_entry[0]+" "+_entry[1]+"\n")
f.close()

f = open(pvdir + "/lmdb/val.txt", "w")
for _entry in VAL_SET:
 f.write(_entry[0]+" "+_entry[1]+"\n")
f.close()


