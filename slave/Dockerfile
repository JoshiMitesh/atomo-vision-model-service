# Slave Node — Person Detection only (see CLAUDE.md).
#
# Build from the REPO ROOT (parent of this slave/ folder), not from
# inside slave/, so the image layout matches how main.py expects to be
# run — "python -m slave.main" from slave/'s parent directory:
#
#   docker build -f slave/Dockerfile -t person-slave .
#
# (docker-compose.yml at the repo root does this for you via
#  build.context: . / build.dockerfile: slave/Dockerfile)

FROM python:3.11-slim

# ffmpeg + the shared libs OpenCV's wheel links against at import time —
# needed for RTSP capture (cv2.VideoCapture with FFMPEG backend) and
# video segment writing (cv2.VideoWriter).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code-only
# changes (ultralytics/torch are large — avoid re-downloading them on
# every rebuild).
COPY slave/requirements.txt slave/requirements.txt
RUN pip install --no-cache-dir -r slave/requirements.txt

COPY slave/ slave/

# Same reasoning as config.py's OMP/OPENBLAS/MKL env vars: keep NumPy's
# and OpenCV's internal thread pools from over-subscribing the CPU
# alongside RTSP decode / NPU driver threads. Belt-and-suspenders with
# config.py's own os.environ.setdefault() calls, in case something
# imports numpy/cv2 before that code runs.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "slave.main"]
