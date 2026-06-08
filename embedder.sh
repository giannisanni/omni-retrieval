#!/usr/bin/env bash
# Load the LCO-Embedding-Omni embedder ONLY when needed.
#
#   Usage: ./embedder.sh start | stop | status
#
# Override paths/port via environment variables:
#   LLAMA_SERVER_BIN  path to the (forked) llama-server binary
#                     default: $HOME/ht-llama.cpp/build/bin/llama-server
#   LCO_MODEL         path to LCO-Embedding-Omni-3B-2605-Q8_0.gguf
#   LCO_MMPROJ        path to mmproj-LCO-Embedding-Omni-3B-2605-F16.gguf
#   LCO_PORT          listen port (default 8090)
#   LCO_LOG           server log path (default $HOME/.lcovec/server.log)

BIN="${LLAMA_SERVER_BIN:-$HOME/ht-llama.cpp/build/bin/llama-server}"
M="${LCO_MODEL:-$HOME/models/lco-omni/LCO-Embedding-Omni-3B-2605-Q8_0.gguf}"
P="${LCO_MMPROJ:-$HOME/models/lco-omni/mmproj-LCO-Embedding-Omni-3B-2605-F16.gguf}"
PORT="${LCO_PORT:-8090}"
LOG="${LCO_LOG:-$HOME/.lcovec/server.log}"
mkdir -p "$(dirname "$LOG")"

case "${1:-}" in
  start)
    if pgrep -f "llama-server.*LCO-Embedding" >/dev/null; then
      echo "already running on :$PORT"; exit 0
    fi
    # The embedder needs ~9 GB of VRAM. Free your GPU first if something else
    # holds it (e.g. `ollama stop <model>`), or lower -ngl to spill to CPU.
    #
    # The two flags below are MANDATORY (see README "Two gotchas"):
    #   LLAMA_MEDIA_MARKER  pins the media placeholder (the fork randomizes it).
    #   --no-mmproj-offload runs the vision/audio projector on CPU (the CUDA CLIP
    #                       graph uses unsupported ops -> degenerate vectors).
    LLAMA_MEDIA_MARKER='<__media__>' nohup "$BIN" -m "$M" --mmproj "$P" --no-mmproj-offload \
      --embedding --pooling last -ngl 99 --host 127.0.0.1 --port "$PORT" -c 8192 \
      > "$LOG" 2>&1 &
    for _ in $(seq 1 120); do
      grep -qi "server is listening" "$LOG" 2>/dev/null && { echo "embedder ready on :$PORT"; exit 0; }
      sleep 2
    done
    echo "timed out waiting for server; see $LOG"; exit 1 ;;
  stop)
    pkill -f "llama-server.*LCO-Embedding" && echo "embedder stopped" || echo "not running" ;;
  status)
    if pgrep -f "llama-server.*LCO-Embedding" >/dev/null; then echo "RUNNING on :$PORT"; else echo "stopped"; fi ;;
  *)
    echo "usage: $0 start|stop|status"; exit 1 ;;
esac
