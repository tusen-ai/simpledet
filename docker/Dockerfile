# modify from https://github.com/NVIDIA/TensorRT/blob/master/docker/ubuntu.Dockerfile
ARG CUDA_VERSION=11.1
ARG CUDNN_VERSION=8
ARG OS_VERSION=16.04
ARG NVCR_SUFFIX=
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION}${NVCR_SUFFIX}

LABEL maintainer="Simpledet"

WORKDIR workspace

# basic
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get install -y build-essential python-dev python3-dev && \
    apt-get install -y git wget sudo curl openssh-server openssh-client bash-completion command-not-found \
    vim htop tmux zsh rsync bzip2 zip unzip patch time make cmake locales locales-all libgtk2.0-dev libgl1-mesa-glx python3-tk \
    ninja-build libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sfv /usr/bin/python3 /usr/bin/python

# zsh and fzf
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    sed -i 's/robbyrussell/fishy/' ~/.zshrc && \
    sed -i 's/(git)/(git zsh-autosuggestions)/' ~/.zshrc && \
    sed -i 's/# DISABLE_AUTO_UPDATE/DISABLE_AUTO_UPDATE/' ~/.zshrc && \
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install --all

# use pyenv to manage python version
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc && \
    echo 'export PYTHON_CONFIGURE_OPTS="--enable-shared"' >> ~/.zshrc && \
    echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc

RUN /usr/bin/zsh -c "source ~/.zshrc && \
                     pyenv install 3.6.8 && \
                     pyenv global 3.6.8 && \
                     eval zsh && \
                     pip install -U pipenv setuptools && \
                     pip install ipython numpy scipy scikit-learn tqdm graphviz easydict matplotlib pyarrow pyzmq pillow cython requests pytz opencv-python tensorboard && \
                     rm -rf ~/.cache"

# build mxnet
RUN /usr/bin/zsh -c "source ~/.zshrc && \
                     git clone --recursive https://github.com/apache/incubator-mxnet /tmp/mxnet -b 1.6.0 && \
                     git clone https://github.com/Tusimple/simpledet /tmp/simpledet && \
                     git clone https://github.com/RogerChern/cocoapi /tmp/cocoapi && \
                     cp -r /tmp/simpledet/operator_cxx/* /tmp/mxnet/src/operator && \
                     mkdir -p /tmp/mxnet/src/coco_api && \
                     cp -r /tmp/cocoapi/common /tmp/mxnet/src/coco_api && \
                     cd /tmp/mxnet && \
                     echo 'USE_SIGNAL_HANDLER = 1' >> ./config.mk && \
                     echo 'USE_OPENCV = 0' >> ./config.mk && \
                     echo 'USE_MKLDNN = 0' >> ./config.mk && \
                     echo 'USE_BLAS = openblas' >> ./config.mk && \
                     echo 'USE_CUDA = 1' >> ./config.mk && \
                     echo 'USE_CUDA_PATH = /usr/local/cuda' >> ./config.mk && \
                     echo 'USE_CUDNN = 1' >> ./config.mk && \
                     echo 'USE_NCCL = 1' >> ./config.mk && \
                     echo 'USE_DIST_KVSTORE = 1' >> ./config.mk && \
                     echo 'CUDA_ARCH = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86' >> ./config.mk && \
                     rm /tmp/mxnet/src/operator/nn/group_norm* && \
                     make -j$((`nproc`-1)) && \
                     cd python && \
                     python3 setup.py install && \
                     rm -rf /tmp/mxnet /tmp/simpledet /tmp/cocoapi"

# install pycocotools and mxnext
RUN /usr/bin/zsh -c "source ~/.zshrc && \
                     pip install 'git+https://github.com/RogerChern/cocoapi.git#subdirectory=PythonAPI' && \
                     pip install 'git+https://github.com/RogerChern/mxnext#egg=mxnext'"

# ssh
RUN chsh -s /usr/bin/zsh root && \
    mkdir /var/run/sshd && \
    echo 'root:simpledet' | chpasswd && \
    sed -i '/PermitRootLogin/s/prohibit-password/yes/' /etc/ssh/sshd_config
EXPOSE 22

# env
RUN echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.zshrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.zsrhc

CMD ["/usr/sbin/sshd", "-D"]

