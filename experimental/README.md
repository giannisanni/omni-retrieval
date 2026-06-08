# Experimental: the transformers (full-fidelity) path

This is **Path B**: serve LCO-Embedding-Omni through HF **transformers** instead of GGUF/llama.cpp. It exists for one reason the GGUF path cannot deliver: **native multi-frame video** (frames + audio fused into a single vector, with real temporal sampling).

> Status: **experimental / reference.** Heavier than the main path, and in our testing its retrieval quality is **below** the GGUF path. Use it for research, native-video work, and development, not as the default. For actually using the tool, see the [main README](../README.md) (Path A).

## Why this exists (and why it is not the default)

The model (Qwen2.5-Omni) supports video; llama.cpp's multimodal stack does not decode video containers, so the main path decomposes video into one frame + audio. This path runs the real Qwen2.5-Omni processor, which samples frames over time and ingests audio, the way the model was meant to consume video.

### What we measured

Two findings from testing on an RTX 4060 Ti (small probes, descriptive):

1. **Retrieval quality is below the GGUF path.** Reproducing LCO's exact embedding in raw transformers is non-trivial: the GGUF/llama.cpp path effectively encodes LCO's pooling/normalization recipe, which the model repo does not document. Our best extraction (mean-pool over post-final-norm hidden states; see `diag.py`) scored **3/4** on the cross-modal text->image probe vs the GGUF path's **4/4**.

2. **Fusing video into one vector did not help retrieval for a speech clip.** On a talking-head clip, the fused native-video vector is dominated by the visual frame tokens (~3,800) and under-weights the audio tokens (~1,500), so it retrieved the clip's *speech* worse than a dedicated audio embedding:

   | query | native_video | frame_only | audio_only |
   |---|---|---|---|
   | speech content of the clip | +0.380 | +0.417 | **+0.706** |

   (Absolute cosines are not comparable across representations due to per-modality anisotropy; the point is that audio clearly separated the speech and the fused video did not.)

**Takeaway:** the main path's two-row decomposition (index the frame and the audio as separate items, calibrated by per-modality z-scoring) preserves the speech signal better than fusing into one native-video vector. Native fused video should win for **motion / temporal / visually-driven** content, not for speech payload. The open problem is matching LCO's true pooling recipe; contributions welcome.

## Install

```bash
# 1. torch matched to your CUDA (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2. the rest
pip install -r requirements-transformers.txt
# 3. the non-GGUF weights (~7.5 GB)
./download_hf_model.sh
```

Needs more VRAM than the GGUF path (loads the full bf16 model). `ffmpeg` is required for `compare.py`.

## Usage

`embed_tf.py` is a library:

```python
from embed_tf import embed_text, embed_image, embed_audio, embed_video
qv = embed_text("a person explaining a workflow")
vv = embed_video("clip.mp4", with_audio=True, fps=1.0)   # native multi-frame + audio
print(float(qv @ vv))
```

Compare native video against frame-only and audio-only for your own clip:

```bash
python compare.py --video clip.mp4 \
    --query "what is said in the clip" \
    --query "what is shown on screen"
```

Inspect the pooling choice:

```bash
python diag.py
```

## Files

| file | purpose |
|---|---|
| `embed_tf.py` | transformers embedder: text / image / audio / native video |
| `compare.py` | native video vs single-frame vs audio-only, for a given clip |
| `diag.py` | extraction-method diagnostic (why mean-pool) |
| `download_hf_model.sh` | fetch the non-GGUF weights |
| `requirements-transformers.txt` | deps (install torch separately) |
