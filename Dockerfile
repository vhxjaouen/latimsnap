# Utilisation d'Ubuntu 24.04 (Noble Numbat)
FROM ubuntu:24.04

# Éviter les questions interactives lors de l'installation des paquets
ENV DEBIAN_FRONTEND=noninteractive

# 1. Installation des dépendances système complètes
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    wget \
    pkg-config \
    libcurl4-openssl-dev \
    # Dépendances Graphiques / X11
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxt-dev \
    libxrender-dev \
    libxcursor-dev \
    libxft-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libvulkan-dev \
    # Dépendances ITK-SNAP spécifiques (vu dans tes logs)
    libssh-dev \
    libsqlite3-dev \
    libexpat1-dev \
    # Paquets Qt6 vérifiés pour 24.04
    qt6-base-dev \
    qt6-base-dev-tools \
    qt6-base-private-dev \
    qt6-declarative-dev \
    qt6-tools-dev \
    qt6-tools-dev-tools \
    libqt6opengl6-dev \
    libqt6svg6-dev \
    # Pour le debugging et l'outil VS Code
    gdb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# 2. Compilation de VTK 9.3.1 (Version avec TOUS les modules SNAP)
RUN git clone --branch v9.3.1 https://github.com/Kitware/VTK.git vtk-src && \
    cmake -S vtk-src -B vtk-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DVTK_GROUP_ENABLE_Qt=YES \
    -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES \
    -DVTK_MODULE_ENABLE_VTK_RenderingQt=YES \
    -DVTK_MODULE_ENABLE_VTK_ViewsQt=YES \
    -DVTK_MODULE_ENABLE_VTK_GUISupportQtQuick=NO \
    -DVTK_MODULE_ENABLE_VTK_RenderingExternal=YES \
    -DVTK_QT_VERSION=6 && \
    cmake --build vtk-build --target install && \
    rm -rf vtk-src vtk-build

# 3. Compilation de ITK 5.4.0 (Inclusion des modules de recherche/segmentation)
RUN git clone --branch v5.4.0 https://github.com/InsightSoftwareConsortium/ITK.git itk-src && \
    cmake -S itk-src -B itk-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DITK_USE_SYSTEM_VTK=ON \
    -DVTK_DIR=/usr/local/lib/cmake/vtk-9.3 \
    # Modules critiques pour ITK-SNAP
    -DModule_ITKReview=ON \
    -DModule_MorphologicalContourInterpolation=ON \
    -DITK_LEGACY_REMOVE=OFF \
    -DITK_BUILD_DEFAULT_MODULES=ON && \
    cmake --build itk-build --target install && \
    rm -rf itk-src itk-build

# 4. Variables d'environnement pour le développement
ENV CXXFLAGS="-fpermissive"
ENV VTK_DIR="/usr/local/lib/cmake/vtk-9.3"
ENV ITK_DIR="/usr/local/lib/cmake/ITK-5.4"

WORKDIR /workspaces/latimsnap
