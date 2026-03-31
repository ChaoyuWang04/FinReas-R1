#!/bin/bash
set -euo pipefail

python3 -m src.training.sft_trainer "$@"
