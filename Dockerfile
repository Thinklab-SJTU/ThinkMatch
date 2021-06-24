FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

LABEL maintainer="runzhong.wang@sjtu.edu.cn"

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get install -y gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN apt-get install -y ninja-build findutils libhdf5-serial-dev git wget libssl-dev

RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz && tar zxvf cmake-3.19.1.tar.gz
RUN cd cmake-3.19.1 && ./bootstrap && make && make install

RUN /opt/conda/bin/pip install tensorboardX scipy easydict pyyaml xlrd xlwt pynvml
RUN /opt/conda/bin/python -m pip install git+https://git@github.com/rogerwwww/lpmp.git

ENV CUDA=cu101 TORCH=1.6.0
RUN /opt/conda/bin/pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN /opt/conda/bin/pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN /opt/conda/bin/pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN /opt/conda/bin/pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN /opt/conda/bin/pip install torch-geometric==1.6.3

RUN update-alternatives --remove gcc /usr/bin/gcc-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
