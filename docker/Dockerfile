FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04

# Install neccessary tools
RUN apt-get update
RUN apt-get install -y \
  software-properties-common \ 
  ca-certificates \
  wget
RUN wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main"
RUN apt-get update
RUN apt-get install -y \ 
    build-essential \
    g++ \
    python-dev \
    autotools-dev \
    libicu-dev \
    libbz2-dev \
    libboost-all-dev 

RUN apt-get install -y  \
    mc \
    lynx \
    libqhull* \
    pkg-config \
    libxmu-dev \
    libxi-dev \
    --no-install-recommends --fix-missing

RUN apt-get install -y  \
    mesa-common-dev \
    vim  \
    git  \
    unzip  \
    mercurial \
    freeglut3-dev \
    libflann-dev \
    --no-install-recommends --fix-missing

RUN apt-get install -y \
    libboost1.58-all-dev \
    libeigen3-dev \
    python \
    libusb-1.0-0-dev \
    libudev-dev \
    doxygen \
    graphviz \
    libpng12-dev \
    libgtest-dev \
    libpcap-dev \
    libvtk5-qt4-dev \
    python-vtk \
    libvtk-java

RUN apt-get install -y \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libjpeg.dev \
    libtiff4.dev \
    libswscale-dev \
    libjasper-dev  

# Install new cmake
RUN cd /opt \
    && wget https://cmake.org/files/v3.9/cmake-3.9.1-Linux-x86_64.tar.gz \
    && tar zxvf cmake-3.9.1-Linux-x86_64.tar.gz \
    && mv cmake-3.9.1-Linux-x86_64 /opt/cmake-3.9.1 \
    && ln -sf /opt/cmake-3.9.1/bin/*  /usr/bin/ 

RUN apt-get autoremove

# Install Eigen
RUN cd /opt \
    && git clone https://github.com/eigenteam/eigen-git-mirror eigen \
    && cd eigen \
    && git checkout tags/3.2.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j 24 \
    && make install

# Install VTK
RUN cd /opt \
    && git clone https://github.com/Kitware/VTK VTK \
    && cd VTK \
    && git checkout tags/v8.0.0 \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE:STRING=Release -D VTK_RENDERING_BACKEND=OpenGL .. \
    && make -j 24 \
    && make install

# Install PCL
RUN cd /opt \
    && wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.0.zip \
    && unzip pcl-1.9.0.zip \
    && cd pcl-pcl-1.9.0 \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=None -D BUILD_GPU=ON -D BUILD_apps=ON -D BUILD_examples=ON .. \
    && make -j 24 \
    && make install

# Install Opencv
RUN cd /opt \
    && wget https://github.com/opencv/opencv/archive/3.4.3.zip \
    && unzip 3.4.3.zip \
    && cd opencv-3.4.3 \
    && mkdir build \
    && cd build \
    && cmake -D WITH_OPENMP=ON -D ENABLE_PRECOMPILED_HEADERS=OFF ..\
    && make -j 24 \
    && make install

# Install gpd
RUN cd /opt \
    && git clone https://github.com/atenpas/gpd gpd \
    && cd gpd \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j24

