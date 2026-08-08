# README_build.md — Building & Packaging ITK-SNAP for Linux

This document explains how the `ITK-SNAP.tar.gz` tarball published from this
repository is produced: how a Qt6 + ITK 5.4 + VTK 9.5 + Python (pTVreg) binary
gets turned into a self-contained archive that runs on essentially any modern
x86_64 Linux with a working OpenGL driver.

The workflow is entirely Docker-based so nothing leaks in from the host system.
The Docker image guarantees a fixed Qt/ITK/VTK stack and glibc floor.

---

## Overview

The pipeline has four stages:

1. **Base image** (`ghcr.io/vhxjaouen/latimsnap-base`) — provides Qt6, ITK 5.4,
   VTK 9.5, CMake, GCC, libssh, libcurl. Long-lived, rebuilt rarely.
2. **Project image** (`latimsnap-builder`) — extends the base with Python and
   the pTVreg runtime deps (`nibabel`, `monai`, `torch`, `scikit-image`,
   `numpy`). Built by `docker build .` in this repo.
3. **Compile** — CMake + `make -j$(nproc)` inside the project image, writing
   into a bind-mounted `build/` directory on the host.
4. **Package** — `make install` to a `DESTDIR` bundle, then `linuxdeployqt` to
   copy every non-system shared library into the bundle and patch the binary's
   RPATH. The result is compressed to `ITK-SNAP.tar.gz`.

Each stage runs inside the container with the working tree bind-mounted, so
build artefacts appear in the host filesystem with the same UID as the user
who launched Docker.

---

## Prerequisites (host)

- Docker (rootless or otherwise; commands assume `sg docker -c` for a user in
  the `docker` group but not logged in with a fresh shell)
- `rclone` configured with a `pcloud:` remote if you want to deploy
- 10 GB free disk (the docker image plus the AppDir bundle plus tarball)
- No Qt / ITK / VTK / Python install required on the host

---

## Stage 1: Build the Docker image

```bash
sg docker -c "docker build -t latimsnap-builder ."
```

The `Dockerfile` in the repo root:

```dockerfile
FROM ghcr.io/vhxjaouen/latimsnap-base:latest

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir \
    nibabel \
    monai \
    torchvision \
    scikit-image \
    numpy

WORKDIR /workspaces/latimsnap
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_X11_NO_MITSHM=1
ENV QT_XCB_GL_INTEGRATION=xcb_glx
ENV QT_QUICK_BACKEND=software
```

The base image `ghcr.io/vhxjaouen/latimsnap-base` is pre-built and pushed to
GHCR. If the pull fails with `401 Unauthorized`, run:

```bash
docker login ghcr.io
```

with a Personal Access Token that has `read:packages` scope.

**Why a separate base image?** ITK and VTK take ~1 hour to compile from source.
Keeping them in a rarely-rebuilt base image means day-to-day project builds are
minutes, not hours.

---

## Stage 2: Compile ITK-SNAP

```bash
sg docker -c 'docker run --rm --entrypoint "" \
  -v "$(pwd):/workspaces/latimsnap" \
  -w /workspaces/latimsnap \
  latimsnap-builder \
  bash -c "mkdir -p build && cd build && cmake .. && make -j$(nproc)"'
```

Key points:

- `--entrypoint ""` overrides any entrypoint in the base image so we run our
  own `bash -c "..."`.
- `-v "$(pwd):/workspaces/latimsnap"` bind-mounts the working tree so build
  output lands directly in the host's `build/` directory. No `docker cp`
  dance.
- `-w /workspaces/latimsnap` puts us in the source root inside the container.
- The `$(nproc)` inside the bash string is evaluated **inside the container**,
  which is what we want — it sees the container's CPU quota. Note the `$` is
  not escaped: `bash -c "..."` with double quotes lets the host shell expand
  `$(pwd)` but the `$(nproc)` sits inside the single-quoted outer string, so
  Docker sees it verbatim.
- The Docker image runs as UID 1000 (the `ubuntu` user in the base image); by
  coincidence host users usually have UID 1000 too, so `build/` files are
  owned by the host user. If your UID differs, add `--user $(id -u):$(id -g)`.

CMake discovers ITK, VTK, and Qt6 from the base image's system paths. There is
no toolchain file. The build produces:

- `build/ITK-SNAP` — the main GUI executable
- `build/Submodules/c3d/*` — c3d command-line tools
- `build/Submodules/greedy/greedy` — greedy registration tool
- `build/itksnap-wt` — the workspace CLI

---

## Stage 3: Install to a bundle directory

```bash
sg docker -c 'docker run --rm --entrypoint "" \
  -v "$(pwd):/workspaces/latimsnap" \
  -w /workspaces/latimsnap \
  latimsnap-builder \
  bash -c "rm -rf build/bundle && cd build && make install DESTDIR=/workspaces/latimsnap/build/bundle"'
```

CMake's install targets copy every executable, resource file, translation, and
the entire `pTVreg/` Python tree into the bundle under
`build/bundle/usr/local/`. Because `DESTDIR` is a path *outside* the normal
install prefix, `make install` produces a self-contained tree we can freely
move around.

Layout after `make install`:

```
build/bundle/usr/local/
├── bin/
│   ├── c3d, c2d, c4d, greedy, ...
│   └── itksnap-wt
└── lib/
    └── snap-4.4.0/
        ├── ITK-SNAP                    ← main binary
        ├── pTVreg/                     ← Python bridge (cli_snap.py etc.)
        └── (linuxdeployqt will drop libs and plugins here)
```

`snap-4.4.0` is the versioned install directory; `RegistrationModel.cxx`
searches for `pTVreg/pTVreg/cli_snap.py` relative to the executable and
`../lib/snap-4.4.0/pTVreg/pTVreg/cli_snap.py` is one of the fallback paths.

---

## Stage 4: Bundle Qt/VTK/ITK libs with linuxdeployqt

At this point `build/bundle/usr/local/lib/snap-4.4.0/ITK-SNAP` runs *only
inside the container*. On a host without ITK/VTK/Qt6 installed, it fails with
missing `.so` errors (`libvtkChartsCore-9.5.so.1: cannot open shared object
file`).

`linuxdeployqt` fixes this by:

1. Walking the binary's `NEEDED` shared-library dependencies recursively.
2. Copying every non-system library into the bundle directory next to the
   binary.
3. Patching the binary's ELF `RPATH` to `$ORIGIN` so the dynamic linker looks
   in the bundle directory first.
4. Copying Qt platform plugins (`platforms/libqxcb.so` etc.) and image format
   plugins.
5. Generating a `qt.conf` so Qt knows where its plugins live.

### 4.1. Download linuxdeployqt

```bash
sg docker -c 'docker run --rm --entrypoint "" \
  -v "$(pwd):/workspaces/latimsnap" \
  -w /workspaces/latimsnap \
  latimsnap-builder \
  bash -c "if [ ! -f linuxdeployqt ]; then \
             wget -c -nv \
               https://github.com/probonopd/linuxdeployqt/releases/download/continuous/linuxdeployqt-continuous-x86_64.AppImage \
               -O linuxdeployqt && chmod a+x linuxdeployqt; fi"'
```

We download the *continuous* build once and cache it in the source tree
(gitignored). It is itself an AppImage.

### 4.2. Extract the AppImage

FUSE is usually not available inside Docker, so the AppImage cannot be executed
directly. AppImages support `--appimage-extract` which unpacks them to a
plain `squashfs-root/` directory that can be run without FUSE:

```bash
sg docker -c 'docker run --rm --entrypoint "" \
  -v "$(pwd):/workspaces/latimsnap" \
  -w /workspaces/latimsnap \
  latimsnap-builder \
  bash -c "if [ ! -d squashfs-root ]; then ./linuxdeployqt --appimage-extract; fi"'
```

Both `linuxdeployqt` and `squashfs-root/` are cached; subsequent builds skip
these steps thanks to the `if [ ! -f ]` / `if [ ! -d ]` guards.

### 4.3. Run linuxdeployqt against the bundled binary

```bash
sg docker -c 'docker run --rm --entrypoint "" \
  -v "$(pwd):/workspaces/latimsnap" \
  -w /workspaces/latimsnap \
  latimsnap-builder \
  bash -c "./squashfs-root/AppRun \
             build/bundle/usr/local/lib/snap-4.4.0/ITK-SNAP \
             -qmake=/usr/bin/qmake6 \
             -bundle-non-qt-libs \
             -unsupported-allow-new-glibc \
           && cd build/bundle/usr/local/lib \
           && tar -czf /workspaces/latimsnap/build/ITK-SNAP.tar.gz snap-4.4.0 \
           && chown 1000:1000 /workspaces/latimsnap/build/ITK-SNAP.tar.gz"'
```

Flag explanations:

- **`build/bundle/usr/local/lib/snap-4.4.0/ITK-SNAP`** — the target binary.
  linuxdeployqt uses its directory as the AppDir root.
- **`-qmake=/usr/bin/qmake6`** — tells linuxdeployqt which Qt to bundle. The
  base image installs Qt6 via distro packages under `/usr`, and `qmake6` is
  the Qt6-specific qmake wrapper.
- **`-bundle-non-qt-libs`** — the crucial one: without this, only Qt libraries
  get bundled and VTK/ITK/GLU/etc. remain as system-only dependencies.
- **`-unsupported-allow-new-glibc`** — linuxdeployqt was designed to target
  the oldest still-supported Ubuntu LTS for maximum portability. The base
  image uses a newer glibc; this flag opts out of the "you're linking against
  a too-new libc" check. The resulting binary will not run on hosts older
  than the build container's glibc, but will run on anything newer.

After linuxdeployqt is done, `build/bundle/usr/local/lib/snap-4.4.0/` contains
the binary, hundreds of `.so` files, and a `plugins/` (or `platforms/`)
subdirectory with Qt plugins. The ELF `RPATH` on the binary points to
`$ORIGIN` so the loader finds the bundled libs first.

### 4.4. Tar and chown

```bash
cd build/bundle/usr/local/lib \
  && tar -czf /workspaces/latimsnap/build/ITK-SNAP.tar.gz snap-4.4.0
```

We `cd` into `.../lib` so the tarball contains `snap-4.4.0/` at the root — the
user extracts it wherever they want and runs `snap-4.4.0/ITK-SNAP`.

`chown 1000:1000` fixes ownership of the tarball to match the host user
(`ubuntu` in the container maps to UID 1000; most Linux users on the host
also have UID 1000).

Typical size: **~110 MB** compressed.

---

## Stage 5: Deploy

```bash
rclone copyto build/ITK-SNAP.tar.gz pcloud:TEMP/ITK-SNAP.tar.gz --progress
```

The `copyto` (as opposed to `copy`) form takes an explicit destination path so
we control the file name on the remote.

---

## Running the tarball on the host

```bash
tar xzf ITK-SNAP.tar.gz
./snap-4.4.0/ITK-SNAP
```

No installation, no `sudo`, no `LD_LIBRARY_PATH` tweaks. The `$ORIGIN` RPATH
handles library discovery. Qt platform plugins are found via the bundled
`qt.conf`.

### Enabling pTVreg deformable registration

The tarball ships `pTVreg/pTVreg/cli_snap.py` under `snap-4.4.0/pTVreg/` but
**does not** ship a Python interpreter or the pTVreg Python dependencies
(`torch`, `nibabel`, `monai`, `scipy`). The user must:

1. Create a Python 3.10+ venv with `torch`, `nibabel`, `monai`, `scipy`,
   `scikit-image`, `numpy`.
2. In the Registration dialog, choose "Deformable (pTVreg)" and browse to the
   venv's `python` executable in the Python (venv) field.

The C++ side (`RegistrationModel::ResolvePTVregScriptPath`) will find
`cli_snap.py` automatically via any of these locations in order:

1. `$SNAP_PTVREG_DIR/pTVreg/cli_snap.py`
2. `<exe-dir>/pTVreg/pTVreg/cli_snap.py`
3. `<exe-dir>/../lib/snap-4.4.0/pTVreg/pTVreg/cli_snap.py` ← *this is where
   `make install` puts it*
4. `<exe-dir>/../share/pTVreg/pTVreg/cli_snap.py`
5. Anywhere within 5 parent directories of the exe (development mode)

---

## Why this approach

Alternatives considered:

| Approach | Pros | Cons |
|---|---|---|
| **AppImage** | Single file, integrated into desktop | FUSE required; extraction step for many CI environments; `--appimage-extract-and-run` inconvenient |
| **Flatpak** | Modern, sandboxed | Overhead of the Flatpak runtime; user must install Flatpak; sandbox breaks Qt file dialogs on X11 |
| **Snap** | Ubuntu-focused, auto-updates | Snap store integration is not desired here; sandboxing is painful |
| **`.deb` package** | Familiar to Ubuntu users | Ties us to specific Ubuntu versions; ITK/VTK conflicts with distro packages |
| **`.tar.gz` bundle** (this) | Distribution-agnostic; no runtime dependency; trivial rollback | User must know how to `tar xzf` and run a binary |

Given ITK-SNAP's user base (imaging researchers on a mix of Linux distros,
often without root), the plain tarball wins on simplicity and portability.

---

## Debugging a broken bundle

If the tarball fails to run:

1. **Check missing libs**:
   ```bash
   ldd snap-4.4.0/ITK-SNAP | grep 'not found'
   ```
2. **Check RPATH**:
   ```bash
   readelf -d snap-4.4.0/ITK-SNAP | grep -E 'RPATH|RUNPATH'
   ```
   Should include `$ORIGIN`.
3. **Check Qt plugin loading**:
   ```bash
   QT_DEBUG_PLUGINS=1 ./snap-4.4.0/ITK-SNAP
   ```
4. **Check pTVreg script resolution**: set `SNAP_PTVREG_DIR=/path/to/pTVreg`
   to override the search.

If a library got copied but the wrong one is picked at runtime, `LD_DEBUG=libs`
gives an exhaustive trace.

---

## Rebuilding faster

- The `linuxdeployqt` step is the slowest (~2 min); the download and AppImage
  extraction only happen the first time.
- Incremental compiles: skip the `rm -rf build/bundle` and let CMake do
  dependency tracking. Only run `make install` + linuxdeployqt when a shared
  library boundary changed.
- The Docker image itself is heavily cached — `docker build` should complete
  in under 5 s once the layers are cached.

For a one-liner that always produces a fresh tarball from a clean bundle:

```bash
./run_build.sh   # in-container compile only
# then the make install + linuxdeployqt + tar sequence above
```

---

## Files gitignored by this pipeline

- `build/` — CMake output, bundle, tarball
- `linuxdeployqt` — the AppImage binary
- `squashfs-root/` — extracted AppImage
- `fake_bin/` — CI stubs (sshpass, curl replacements)
- `pTVreg/runs/`, `pTVreg/playground/`, `pTVreg/matlab/`, `__pycache__/`,
  `*.egg-info/` — pTVreg run outputs and generated caches
