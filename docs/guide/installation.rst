=============
Installation
=============
Docker
-----------
We recommend using docker images if docker_ or other container runtimes e.g. singularity_ is available on your devices.

1. We maintain a prebuilt image at dockerhub_:
    ::

        runzhongwang/thinkmatch:torch1.6.0-cuda10.1-cudnn7-pyg1.6.3

    It can be used by docker or other container runtimes that support docker images e.g. singularity_.
#. We also provide a ``Dockerfile`` to build your own image (you may need ``docker`` and ``nvidia-docker`` installed on your computer).

.. _docker: https://www.docker.com/
.. _dockerhub: https://hub.docker.com/r/runzhongwang/thinkmatch
.. _singularity: https://sylabs.io/singularity/

Manual configuration
--------------------------

This repository is developed and tested with Ubuntu 16.04, Python 3.7, Pytorch 1.6, cuda10.1, cudnn7 and torch-geometric 1.6.3.
If docker is not available, we provide detailed steps to install the requirements by ``apt`` and ``pip``.

1. Install and configure Pytorch 1.6 (with GPU support). Please follow the `official guidelines <https://pytorch.org/get-started/locally/>`_.
#. Install ninja-build:
    ::

        apt-get install ninja-build

#. Install python packages:
    ::

        pip install tensorboardX scipy easydict pyyaml xlrd xlwt pynvml

#. Install building tools for LPMP:
    ::

        apt-get install -y findutils libhdf5-serial-dev git wget libssl-dev

        wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz && tar zxvf cmake-3.19.1.tar.gz
        cd cmake-3.19.1 && ./bootstrap && make && make install

#. Install and build LPMP:
    ::

        python -m pip install git+https://git@github.com/rogerwwww/lpmp.git

    You may need ``gcc-9`` to successfully build LPMP. Here we provide an example installing and configuring ``gcc-9``:
    ::

       apt-get update
       apt-get install -y software-properties-common
       add-apt-repository ppa:ubuntu-toolchain-r/test

       apt-get install -y gcc-9 g++-9
       update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

#. Install torch-geometric:
    ::

        export CUDA=cu101
        export TORCH=1.6.0
        /opt/conda/bin/pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
        /opt/conda/bin/pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
        /opt/conda/bin/pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
        /opt/conda/bin/pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
        /opt/conda/bin/pip install torch-geometric==1.6.3

#. If you have configured ``gcc-9`` to build LPMP, be sure to switch back to ``gcc-7`` because ThinkMatch is based on ``gcc-7``. Here is also an example:
    ::

        update-alternatives --remove gcc /usr/bin/gcc-9
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7

