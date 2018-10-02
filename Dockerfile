FROM anibali/pytorch:no-cuda


# Prefetch ai2thor data
RUN sudo apt-get update && sudo apt-get -y install wget unzip \
    && mkdir /app/data \
    && cd /app/data \
    && rm -f *.h5 \ 
    && wget http://vision.stanford.edu/yukezhu/thor_v1_scene_dumps.zip \
    && unzip thor_v1_scene_dumps.zip \
    && rm thor_v1_scene_dumps.zip \
    && cd /app

# Prefetch: install packages to previous layers
RUN python -m pip install matplotlib ai2thor Cython scikit-image

COPY . /app
RUN python -m pip install -r /app/requirements.txt