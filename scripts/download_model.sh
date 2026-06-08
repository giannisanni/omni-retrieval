#!/usr/bin/env bash
# Download the LCO-Embedding-Omni-3B-2605 GGUF weights (Q8_0) and the F16
# multimodal projector into ~/models/lco-omni (override with $1).
set -e
DEST="${1:-$HOME/models/lco-omni}"
BASE="https://huggingface.co/marksverdhei/LCO-Embedding-Omni-3B-2605-GGUF/resolve/main"
mkdir -p "$DEST"; cd "$DEST"

echo "downloading Q8_0 weights (~3.4 GB) ..."
wget -c -q --show-progress "$BASE/LCO-Embedding-Omni-3B-2605-Q8_0.gguf"
echo "downloading F16 mmproj (~2.5 GB) ..."
wget -c -q --show-progress "$BASE/mmproj-LCO-Embedding-Omni-3B-2605-F16.gguf"
echo "saved to $DEST"
