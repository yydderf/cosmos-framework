  docker run \
    -it \
    -e HF_TOKEN=$HF_TOKEN -e HF_HUB_OFFLINE=1 \
    --name cosmos3_policy_server \
    --net host \
    --rm \
    --runtime nvidia \
    --cap-add=SYS_ADMIN \
    -v /opt/nvidia/nsight-systems/2026.4.1/target-linux-x64:/opt/nsight/target-linux-x64:ro \
    -v .:/workspace \
    -v /workspace/.venv \
    -v /srv/models/huggingface:/root/.cache/huggingface \
    -v $HOME/.cache/uv:/root/.cache/uv \
    --entrypoint /workspace/deploy/workstation/entrypoint.sh \
    cosmos-framework:latest \
    bash -c '\
      uv sync \
        --all-extras \
        --group=cu130-torch213-train \
        --group=policy-server && \
      exec bash; \
    '
