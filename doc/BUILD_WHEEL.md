## Introduction
This document describes the process of packaging our custom mxnet as a python wheel for local installation.

### Platform
In order to suppport CentOS 7, which is popular in production, we have to make the wheel largely comformant to `manylinux2014` tag of pypi.
The `manylinux2014` tag specify the maximum version of some system library as
```
GLIBC_2.17
CXXABI_1.3.7
GLIBCXX_3.4.19
GCC_4.8.5
```
Compiling the library under such restriction from a newer platform could be quite tricky, since this is essential cross-compiling.
So here we complile from the Ubuntu 14.04 which is also `manylinux2014` comformant.

### Setup toolchains
```bash
sudo apt-get update && \
sudo apt-get install -y git \
    vim \
    libcurl4-openssl-dev \
    unzip \
    gcc-4.8 \
    g++-4.8 \
    gfortran \
    gfortran-4.8 \
    binutils \
    nasm \
    libtool \
    curl \
    wget \
    sudo \
    gnupg \
    gnupg2 \
    gnupg-agent \
    pandoc \
    python3-pip \
    automake \
    pkg-config

wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz && \
    tar xzf cmake-3.15.2.tar.gz && \
    cd cmake-3.15.2 && \
    ./configure && make install -j4 && cd .. && \
    rm -r cmake-3.15.2 cmake-3.15.2.tar.gz

# change the url to your repo link if you are doing PR
export SIMPLEDET_URL=https://github.com/tusimple/simpledet
git clone --recursive --depth=1 https://github.com/apache/incubator-mxnet /work/mxnet && \
    cd /work/mxnet && \
    git clone $SIMPLEDET_URL /work/simpledet && \
    cp -r /work/simpledet/operator_cxx/* /work/mxnet/src/operator && \
    git clone https://github.com/RogerChern/cocoapi /work/cocoapi && \
    mkdir -p src/coco_api && \
    cp -r /work/cocoapi/common src/coco_api && \
    rm /work/mxnet/src/operator/nn/group_norm* && \
    rm -r /work/cocoapi /work/simpledet
```

### Compile `libmxnet.so` with static dependancy
```
cd /work/mxnet
# remove sm_30
sed -i 's/KNOWN_CUDA_ARCHS :=.*/KNOWN_CUDA_ARCHS := 35 50 60 70/' Makefile
# change build config according to the target CUDA version
tools/staticbuild/build.sh cu100 pip
# tools/staticbuild/build.sh cu101 pip
```

### Package wheel
```
cd /work/mxnet/tools/pip
ln -s /work/mxnet mxnet-build

# change the path according to the target CUDA version
LD_LIBRARY_PATH=/work/mxnet/staticdeps/usr/local/cuda-10.0/lib64:/work/mxnet/staticdeps/usr/lib/x86_64-linux-gnu:/work/mxnet/staticdeps/usr/lib/nvidia-410
# LD_LIBRARY_PATH=/work/mxnet/staticdeps/usr/local/cuda-10.1/lib64:/work/mxnet/staticdeps/usr/lib/x86_64-linux-gnu:/work/mxnet/staticdeps/usr/lib/nvidia-418
export LD_LIBRARY_PATH
mxnet_variant=CU100 python3 setup.py bdist_wheel
# mxnet_variant=CU101 python3 setup.py bdist_wheel
```

The built wheel file is in `dist/`
