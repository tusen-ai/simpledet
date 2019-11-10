#!/bin/bash


# install dependency
sudo apt update && sudo apt install -y git wget make python3-dev libglib2.0-0 libsm6 libxext6 libxrender-dev unzip

# create conda env
conda create -n simpledet python=3.7
conda activate simpledet

# fetch CUDA environment
conda install cudatoolkit=10.1

# install python dependency
pip install 'matplotlib<3.1' opencv-python pytz

# download and intall pre-built wheel for CUDA 10.1
pip install https://1dv.alarge.space/mxnet_cu101-1.6.0b20190820-py2.py3-none-manylinux1_x86_64.whl

# install pycocotools
pip install 'git+https://github.com/RogerChern/cocoapi.git#subdirectory=PythonAPI'

# install mxnext, a wrapper around MXNet symbolic API
pip install 'git+https://github.com/RogerChern/mxnext#egg=mxnext'

# get simpledet
git clone https://github.com/tusimple/simpledet
cd simpledet
make

# make data dir
mkdir -p data/coco/images data/src

# skip this if you have the zip files
wget -c http://images.cocodataset.org/zips/train2017.zip -O data/src/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip -O data/src/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip -O data/src/test2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/src/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip -O data/src/image_info_test2017.zip

unzip data/src/train2017.zip -d data/coco/images
unzip data/src/val2017.zip -d data/coco/images
unzip data/src/test2017.zip -d data/coco/images
unzip data/src/annotations_trainval2017.zip -d data/coco
unzip data/src/image_info_test2017.zip -d data/coco

python utils/create_coco_roidb.py --dataset coco --dataset-split train2017
python utils/create_coco_roidb.py --dataset coco --dataset-split val2017
python utils/create_coco_roidb.py --dataset coco --dataset-split test-dev2017
