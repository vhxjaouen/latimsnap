FROM ghcr.io/vhxjaouen/latimsnap-base:latest

# 2. Install Python, Pip, and dependencies for Torch/Nibabel
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install the heavy hitters
# Note: We use --no-cache-dir to keep the image size down
RUN pip3 install --break-system-packages --no-cache-dir \
    nibabel \
    monai \
    torchvision \
    scikit-image \
    numpy 

# 2. Set the working directory (standard for VS Code)
WORKDIR /workspaces/latimsnap

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_X11_NO_MITSHM=1
ENV QT_XCB_GL_INTEGRATION=xcb_glx
ENV QT_QUICK_BACKEND=software

# RUN apt-get update && apt-get install -y some-tool
