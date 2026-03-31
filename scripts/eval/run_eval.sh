#!/bin/bash
set -euo pipefail

python3 -m src.eval.benchmark "$@"
