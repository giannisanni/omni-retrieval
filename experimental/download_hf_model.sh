#!/usr/bin/env bash
# Download the non-GGUF LCO-Embedding-Omni-3B-2605 weights (~7.5 GB safetensors)
# into ~/models/lco-omni-hf (override with $1). Used by the transformers path.
set -e
DEST="${1:-$HOME/models/lco-omni-hf}"
BASE="https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-3B-2605/resolve/main"
mkdir -p "$DEST"; cd "$DEST"

FILES="config.json generation_config.json preprocessor_config.json tokenizer_config.json \
chat_template.json special_tokens_map.json tokenizer.json vocab.json merges.txt \
added_tokens.json model.safetensors.index.json \
model-00001-of-00002.safetensors model-00002-of-00002.safetensors"

for f in $FILES; do
  echo "downloading $f ..."
  wget -c -q --show-progress "$BASE/$f" -O "$f"
done
echo "saved to $DEST"
