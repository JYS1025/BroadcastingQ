#!/bin/bash
# Automatically move to the project root directory regardless of where the script is executed
cd "$(dirname "$0")/../../" || exit

python main.py --config applications/network_routing/config_qlearning.yaml
python main.py --config applications/network_routing/config_dqn.yaml
python main.py --config applications/network_routing/config_sbq.yaml
