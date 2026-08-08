# AI Agent Instructions for ITK-SNAP Build and Deployment

When asked to build, test, and deploy ITK-SNAP for Linux, follow this strict procedure using the provided Docker environment. This ensures all Qt/VTK dependencies are properly bundled and the resulting binary runs successfully on the local machine without missing shared object errors (e.g., `libvtkChartsCore-9.3.so.1`).

## 1. Build the Docker Image
Ensure the Docker compilation environment is up to date:
```bash
sg docker -c "docker build -t latimsnap-builder ."
```
*(Note: If pulling the base image fails with a 401 Unauthorized error, remind the user to run `docker login ghcr.io` with a PAT that has `read:packages` scope).*

## 2. Compile the Application
Run the build inside the container, mapping the current directory as a volume.
```bash
sg docker -c 'docker run --rm --entrypoint "" -v "$(pwd):/workspaces/latimsnap" -w /workspaces/latimsnap latimsnap-builder bash -c "mkdir -p build && cd build && cmake .. && make -j\$(nproc)"'
```

## 3. Package the Executable (linuxdeployqt)
Because the binary dynamically links against libraries inside the container, it must be bundled.
```bash
# 3.1 Install the project into a bundle folder
sg docker -c 'docker run --rm --entrypoint "" -v "$(pwd):/workspaces/latimsnap" -w /workspaces/latimsnap latimsnap-builder bash -c "rm -rf build/bundle && cd build && make install DESTDIR=/workspaces/latimsnap/build/bundle"'

# 3.2 Ensure linuxdeployqt is available (Download if missing)
sg docker -c 'docker run --rm --entrypoint "" -v "$(pwd):/workspaces/latimsnap" -w /workspaces/latimsnap latimsnap-builder bash -c "if [ ! -f linuxdeployqt ]; then wget -c -nv https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage -O linuxdeployqt && chmod a+x linuxdeployqt; fi"'

# 3.3 Extract linuxdeployqt AppImage (since FUSE might not be available in docker)
sg docker -c 'docker run --rm --entrypoint "" -v "$(pwd):/workspaces/latimsnap" -w /workspaces/latimsnap latimsnap-builder bash -c "if [ ! -d squashfs-root ]; then ./linuxdeployqt --appimage-extract; fi"'

# 3.4 Run linuxdeployqt to bundle dependencies and patch RPATH, then compress
sg docker -c 'docker run --rm --entrypoint "" -v "$(pwd):/workspaces/latimsnap" -w /workspaces/latimsnap latimsnap-builder bash -c "./squashfs-root/AppRun build/bundle/usr/local/lib/snap-4.4.0/ITK-SNAP -qmake=/usr/bin/qmake6 -bundle-non-qt-libs -unsupported-allow-new-glibc && cd build/bundle/usr/local/lib && tar -czf /workspaces/latimsnap/build/ITK-SNAP.tar.gz snap-4.4.0 && chown 1000:1000 /workspaces/latimsnap/build/ITK-SNAP.tar.gz"'
```

## 4. Deploy via Rclone
Upload the self-contained package to pCloud using `rclone`.
```bash
rclone copyto build/ITK-SNAP.tar.gz pcloud:TEMP/ITK-SNAP.tar.gz --progress
```
