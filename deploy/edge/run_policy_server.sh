#!/bin/bash
# python -m cosmos_framework.scripts.action_policy_server_robolab \
#   --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
#   --port 8000 \
#   --format-prompt-as-json True \
#   --guidance-interval 960 1001 \
#   --num-steps 10
#
# RTX3090 graphics processor uses the GA102 architecture (--gpu-metrics-set=ga102)
COSMOS_DIST_BACKEND=gloo \
MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '-') \
POLICY=cosmos3 DATETIME=$(date +"%Y%m%d_%H%M") \
    nsys profile \
    --output /workspace/outputs/nsys/%q{MODEL}_%q{POLICY}_%q{DATETIME}_%p --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=ga10y \
    --gpu-metrics-frequency=2000 \
    --cuda-graph-trace=node \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi --capture-range-end=repeat:2 \
    --sample=process-tree --cpuctxsw=process-tree \
    python -m cosmos_framework.scripts.action_policy_server_robolab \
        --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
        --port 8000 \
        --format-prompt-as-json True \
        --guidance-interval 960 1001 \
        --num-steps 4
