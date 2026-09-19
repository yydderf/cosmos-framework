  docker run \
    -it \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    --net host \
    --rm \
    --runtime nvidia \
    --cap-add=SYS_ADMIN \
    -v /opt/nvidia/nsight-systems/2026.4.1/target-linux-x64:/opt/nsight/target-linux-x64:ro \
    -v .:/workspace \
    -v /workspace/.venv \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.cache/uv:/root/.cache/uv \
    cosmos-framework:latest \
    bash -c '\
      uv sync \
        --all-extras \
        --group=cu130-torch213-train \
        --group=policy-server && \
      exec bash; \
    '
