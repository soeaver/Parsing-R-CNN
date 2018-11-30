#!/usr/bin/env bash

cd ../../
if [ ! -d "./data" ]; then
  mkdir data
  echo 'Make data folder!'
fi

cd data
if [ ! -d "./coco" ]; then
  mkdir coco
  echo 'Make coco folder!'
fi

cd coco
if [ ! -d "./images" ]; then
  mkdir images
  echo 'Make images folder!'
fi

cd ../

COCOPATH=/home/user/Database/MSCOCO2017

if [ ! -L "./coco/annotations" ]; then
  if [ -d $COCOPATH/original_annotations ]; then
    ln -s $COCOPATH/original_annotations ./coco/annotations
  else
    ln -s $COCOPATH/annotations ./coco/annotations
  fi
  echo 'Make annotations soft link!'
fi

if [ ! -L "./coco/images/train2017" ]; then
  ln -s $COCOPATH/train2017 ./coco/images/train2017
  echo 'Make train2017 soft link!'
fi

if [ ! -L "./coco/images/val2017" ]; then
  ln -s $COCOPATH/val2017 ./coco/images/val2017
  echo 'Make val2017 soft link!'
fi

if [ ! -L "./coco/images/test2017" ]; then
  ln -s $COCOPATH/test2017 ./coco/images/test2017
  echo 'Make test2017 soft link!'
fi

echo "Done!"
