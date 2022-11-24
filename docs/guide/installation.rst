=============
Installation
=============
Docker
-----------
We recommend using docker images if docker_ or other container runtimes e.g. singularity_ is available on your devices.

We maintain prebuilt images at DockerHub_:

.. warning::

    This doc may not be up-to-date and there might be newer versions available. Please checkout DockerHub_ and install
    the latest version!

::

    runzhongwang/thinkmatch:torch1.6.0-cuda10.1-cudnn7-pyg1.6.3-pygmtools0.3.2

It can be used by docker or other container runtimes that support docker images e.g. singularity_. If you are using
docker, run the following command to pull the image:

::

    docker pull runzhongwang/thinkmatch:torch1.6.0-cuda10.1-cudnn7-pyg1.6.3-pygmtools0.3.2


Other docker images with different ``torch/cuda/pyg`` combinations are also provided to fit the needs of various GPU devices.
Please check from the Internet which CUDA version best suits your GPU.

::

    runzhongwang/thinkmatch:torch1.6.0-cuda10.1-cudnn7-pyg1.6.3-pygmtools0.3.2 # recommended for GTX10 and RTX20 GPUs
    runzhongwang/thinkmatch:torch1.7.1-cuda11.0-cudnn8-pyg1.6.3-pygmtools0.3.2
    runzhongwang/thinkmatch:torch1.10.0-cuda11.3-cudnn8-pyg2.0.3-pygmtools0.3.2 # recommended for RTX30 GPUs

.. note::

    It is recommended to use the ``torch1.6.0-cuda10.1`` image if it is compatible with your devices, because our code
    are developed and tested on this image. If you are encountering any issues on other images, we may not guarantee
    to quickly resolve them because we do not have the same devices as yours.

.. note::

    If you find the existing image does not suit your need and you want a new image, please raise an issue and provide
    the ``torch/cuda/cudnn/pyg/pygmtools`` versions you required.

For more information about the docker images, please check out ThinkMatch-runtime_

.. _docker: https://www.docker.com/
.. _DockerHub: https://hub.docker.com/r/runzhongwang/thinkmatch/tags
.. _singularity: https://sylabs.io/singularity/
.. _ThinkMatch-runtime: https://github.com/Thinklab-SJTU/ThinkMatch-runtime

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

        pip install tensorboardX scipy easydict pyyaml xlrd xlwt pynvml pygmtools

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
