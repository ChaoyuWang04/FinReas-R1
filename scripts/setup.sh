#!/bin/bash
set -euo pipefail

echo "Create your Python environment first, then install baseline dependencies:"
echo "  pip install -r requirements.txt"
echo
echo "Pinned external dependencies used by the original training flow:"
echo "  verl commit: e49fb572bf85a8f0ef7124c898f509bd6d9832a1"
echo "  vllm commit: ed6e9075d31e32c8548b480a47d1ffb77da1f54c"
echo "  flash-attn:  2.7.2.post1"
echo
echo "Clone external projects under dependencies/ as needed."
