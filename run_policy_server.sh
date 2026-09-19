python -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
  --port 8000 \
  --format-prompt-as-json True \
  --guidance-interval 960 1001 \
  --num-steps 10
