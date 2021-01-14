# Docker
## Build
```
cd docker
# build for cuda11.1 cudnn8
docker build --network=host --build-arg OS_VERSION=16.04 --build-arg CUDA_VERSION=11.1 --build-arg CUDNN_VERSION=8 --tag simpledet .
```

## Launch
```
docker run -it --gpus all simpledet zsh
```
