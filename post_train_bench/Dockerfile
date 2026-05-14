FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN chmod 1777 /tmp && \
    apt-get update && apt-get install -y \
        software-properties-common git wget curl build-essential uuid-runtime \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Node.js 22.x
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

RUN npm install -g @openai/codex@0.125.0

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install torch first to anchor the CUDA version.
RUN uv pip install --system --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cu128

# vLLM, resolving against the installed torch.
RUN uv pip install --system --no-cache vllm==0.11.0

RUN uv pip install --system --no-cache ninja==1.13.0 packaging==26.0

# ML packages pinned to match the PostTrainBench eval container family.
RUN uv pip install --system --no-cache \
    accelerate==1.12.0 \
    boto3==1.40.61 \
    bitsandbytes==0.49.1 \
    datasets==4.5.0 \
    evaluate==0.4.6 \
    lm-eval==0.4.10 \
    openai==2.17.0 \
    pandas==2.2.3 \
    scikit-learn==1.7.2 \
    shortuuid==1.0.13 \
    tokenizers==0.22.2 \
    transformers==4.57.3 \
    trl==0.27.2 \
    peft==0.18.1 \
    tiktoken==0.12.0 \
    inspect-ai==0.3.150 \
    matplotlib==3.10.8 \
    certifi==2026.1.4 \
    huggingface-hub==0.36.0

RUN uv pip install --system --no-cache wheel setuptools einops psutil && \
    uv pip install --system --no-cache flash_attn==2.8.3 --no-build-isolation

# inspect_evals pinned to the PostTrainBench eval container commit.
RUN cd /opt && \
    git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd inspect_evals && \
    git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 && \
    uv pip install --system --no-cache .

ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"
