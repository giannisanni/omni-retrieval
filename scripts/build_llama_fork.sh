#!/usr/bin/env bash
# Clone and build the ht-llama.cpp fork that adds the Qwen2.5-Omni embedding
# arch + QWEN25O projector needed by LCO-Embedding-Omni. Mainline llama.cpp and
# Ollama will NOT load this model's multimodal embedding path.
#
# Requires: git, cmake, a CUDA toolkit (for GPU). Adjust CMAKE_CUDA_ARCHITECTURES
# to your GPU (89 = Ada / RTX 40-series; 86 = Ampere; 75 = Turing).
set -e
DEST="${1:-$HOME/ht-llama.cpp}"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-89}"

if [ ! -d "$DEST" ]; then
  git clone --depth 1 https://github.com/heiervang-technologies/ht-llama.cpp "$DEST"
fi
cd "$DEST"
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build build --target llama-server llama-embedding llama-mtmd-cli -j"$(nproc)"
echo "built: $DEST/build/bin/llama-server"
