## Install

```
# install pytorch 1.1 and torchvision
sudo pip3 install torch==1.1 torchvision

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
sudo python setup.py install --cuda_ext --cpp_ext

# clone Hier-R-CNN
git clone https://github.com/soeaver/Hier-R-CNN.git

# install other requirements
pip3 install -r requirements.txt

# mask ops
cd Hier-R-CNN
sh make.sh

# make cocoapi
cd Hier-R-CNN/cocoapi/PythonAPI
mask
cd ../../
ln -s cocoapi/PythonAPI/pycocotools/ ./
```

## Data and Pre-train weights

  Make sure to put the files as the following structure:

  ```
  ├─data
  │  ├─coco
  │  │  ├─images
  │  │  │  ├─train2017
  │  │  │  ├─val2017
  │  │  ├─annotations
  │  │  │  ├─COCOHumanParts
  │  │  │  │  ├─person_humanparts_train2017.json
  │  │  │  │  ├─person_humanparts_val2017.json
  │
  ├─weights
     ├─resnet50_caffe.pth
     ├─resnet101_caffe.pth
     ├─resnext101_32x8d-8ba56ff5.pth

  ```
  

