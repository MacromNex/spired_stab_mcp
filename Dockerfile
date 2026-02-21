FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model weights
COPY scripts/ ./scripts/
RUN chmod -R a+r /app/scripts/
COPY src/ ./src/
RUN chmod -R a+r /app/src/
COPY examples/ ./examples/
RUN chmod -R a+r /app/examples/

# Create directories for jobs and temporary files
RUN mkdir -p jobs tmp

# Pre-download ESM-2 models into the image so they are cached at runtime.
# esm2_t33_650M_UR50D (~2.3GB) and esm2_t36_3B_UR50D (~11GB) are fetched
# via torch.hub and stored under /root/.cache/torch/hub/.
RUN python -c "\
import torch; \
torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D'); \
torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D'); \
print('ESM-2 models cached successfully')"

ENV PYTHONPATH=/app

CMD ["python", "src/server.py"]
