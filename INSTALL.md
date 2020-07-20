## Install

```
# install pytorch 1.1 and torchvision
sudo pip3 install torch==1.1 torchvision==0.3.0

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
pip3 install -v --no-cache-dir ./

# clone Hier-R-CNN
git clone https://github.com/soeaver/Hier-R-CNN.git

# install other requirements
pip3 install -r requirements.txt

# mask ops
cd Hier-R-CNN
sh make.sh

# make cocoapi
cd Hier-R-CNN/cocoapi/PythonAPI
make
cd ../../../
ln -s Hier-R-CNN/cocoapi/PythonAPI/pycocotools/ ./
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
  │  │  │  ├─DensePoseData
  │  │  │  │  ├─densepose_coco_train2017.json
  │  │  │  │  ├─densepose_coco_val2017.json
  │  │  │  │  ├─densepose_coco_test2017.json
  |  |
  │  ├─CIHP
  │  │  ├─train_img
  │  │  │─train_parsing
  │  │  │─train_seg
  │  │  ├─val_img
  │  │  │─val_parsing
  │  │  │─val_seg  
  │  │  ├─annotations
  │  │  │  ├─CIHP_train.json
  │  │  │  ├─CIHP_val.json
  |  |
  │  ├─MHP-v2
  │  │  ├─train_img
  │  │  │─train_parsing
  │  │  │─train_seg
  │  │  ├─val_img
  │  │  │─val_parsing
  │  │  │─val_seg  
  │  │  ├─annotations
  │  │  │  ├─MHP-v2_train.json
  │  │  │  ├─MHP-v2_val.json
  |
  ├─weights
     ├─resnet50_caffe.pth
     ├─resnet101_caffe.pth
     ├─resnext101_32x8d-8ba56ff5.pth

  ```
  
  - Densepose estimation using original coco images.
  - For training and evaluating densepose estimation on Parsing R-CNN, you need fetch DensePose data following [original repo](https://github.com/facebookresearch/DensePose/blob/master/INSTALL.md#fetch-densepose-data)

