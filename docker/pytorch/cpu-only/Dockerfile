# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
#
# This Dockerfile will build Docker Image with PyTorch+MKLDNN+BLIS and Torchvision installed for CPU only
#
# BLIS_URL can be provided by user while building the Docker Image (By Default, it is set to FLAME BLIS git repo)
# Example commandline to build PyTorch with AMD BLIS
# DOCKER_BUILDKIT=1 docker build --build-arg BLIS_URL=https://github.com/amd/blis.git . -t Docker-Image-Repo-Name
ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.7

FROM ${BASE_IMAGE} as dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
RUN curl -v -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build && \
    /opt/conda/bin/conda install -y nomkl pyyaml numpy ipython ninja setuptools cmake cffi typing future && \
    /opt/conda/bin/conda clean -ya

ARG BLIS_URL=https://github.com/flame/blis.git
#Download BLIS and place the necessary library and include files at /usr/local/lib and /usr/local/include respectively
RUN cd ~ && git clone ${BLIS_URL} &&\
    cd ~/blis && ./configure --prefix=~/BLISBuild --enable-cblas --enable-threading=openmp auto && make -j && make install && \
    cp ~/BLISBuild/lib/* /usr/local/lib/ && cp -r ~/BLISBuild/include/blis /usr/local/include/ && \
    if [ ! -e /usr/local/lib/libblis.so ] ; then cp /usr/local/lib/libblis*.so /usr/local/lib/libblis.so ; fi

#Build PyTorch with MKLDNN+BLIS (CPU only)
RUN cd ~ && git clone https://github.com/ROCmSoftwarePlatform/pytorch.git
RUN cd ~/pytorch && git submodule update --init --recursive
RUN cd ~/pytorch && export BLAS=blis USE_MKLDNN_CBLAS=ON WITH_BLAS=blis && USE_ROCM=0 python setup.py install

#Build Torchvision for CPU only
RUN cd ~ && git clone https://github.com/pytorch/vision.git
RUN cd ~/vision && USE_ROCM=0 python setup.py install

