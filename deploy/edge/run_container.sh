#!/bin/bash

docker run -it --rm \
  --name cosmos3_policy_server \
  --net host --runtime nvidia --cap-add=SYS_ADMIN \
  -e HF_TOKEN=$HF_TOKEN \
  -e NSYS_ROOT=/opt/nvidia/nsight-systems/2026.3.1 \
  -v /opt/nvidia/nsight-systems/2026.3.1:/opt/nvidia/nsight-systems/2026.3.1:ro \
  -v .:/workspace \
  -v /workspace/.venv \
  -v uv-cache:/root/.cache/uv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /workspace/deploy/edge/entrypoint.sh \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --inexact --all-extras \
      --group=cu130-torch213-train \
      --group=policy-server && \
      exec bash; \
  '
