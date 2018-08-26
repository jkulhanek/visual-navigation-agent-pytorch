FROM anibali/pytorch:no-cuda

COPY . /app
RUN /app/setup.py install