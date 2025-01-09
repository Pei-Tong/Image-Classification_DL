FROM python:3.12

# Set environment variable （decrease TensorFlow warning）
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_XLA_FLAGS=--tf_xla_enable_xla_devices=0

# Set work directory
WORKDIR /app

# copy project file
COPY . .

# install Python package
RUN pip install --no-cache-dir -r requirements.txt

# user can decide to use Flask or Streamlit
CMD ["bash"]
