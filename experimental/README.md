# Experimental: the transformers (full-fidelity) path

This is **Path B**: serve LCO-Embedding-Omni through HF **transformers** instead of GGUF/llama.cpp. It exists for one reason the GGUF path cannot deliver: **native multi-frame video** (frames + audio fused into a single vector, with real temporal sampling).

> Status: **experimental / reference.** Heavier than the main path, and in our testing its retrieval quality is **below** the GGUF path. Use it for research, native-video work, and development, not as the default. For actually using the tool, see the [main README](../README.md) (Path A).

## Why this exists (and why it is not the default)

The model (Qwen2.5-Omni) supports video; llama.cpp's multimodal stack does not decode video containers, so the main path decomposes video into one frame + audio. This path runs the real Qwen2.5-Omni processor, which samples frames over time and ingests audio, the way the model was meant to consume video.

### What we measured

Findings from testing on an RTX 4060 Ti (small probes, descriptive):

1. **LCO's pooling recipe, reverse-engineered by matching the GGUF output (`match_recipe.py`).** The embedding is the **last token of the final decoder layer's pre-final-norm hidden state** (`hidden_states[-1]`), L2-normalized. On text it reproduces the GGUF vectors at **mean cosine 0.9985** (vs 0.96 for post-norm last-token, 0.52 for mean-pooling - the model repo documents none of this). `embed_tf.py` now uses this recipe.

2. **Text matches GGUF exactly; images are limited by preprocessing, not pooling.** With the matched recipe, transformers text embeddings equal GGUF (0.9985), but image embeddings reach only ~0.81 cosine to GGUF and score **3/4** on the cross-modal probe (vs GGUF's 4/4). The gap is **image preprocessing** - llama.cpp's CLIP resize/normalization differs from the HF processor - which changes the image features before pooling. Matching that pixel pipeline is the remaining open problem.

3. **Fusing video into one vector did not help retrieval for a speech clip.** On a talking-head clip, the fused native-video vector is dominated by the visual frame tokens (~3,800) and under-weights the audio tokens (~1,500), so it retrieved the clip's *speech* worse than a dedicated audio embedding:

   | query | native_video | frame_only | audio_only |
   |---|---|---|---|
   | speech content of the clip | +0.380 | +0.417 | **+0.706** |

   (Absolute cosines are not comparable across representations due to per-modality anisotropy; the point is that audio clearly separated the speech and the fused video did not.)

**Takeaway:** the pooling recipe is solved (text-exact); native video helps for **motion / visually-driven** content, not speech payload. The main GGUF path fuses video with fewer frames + per-item calibration so a lone video stays findable (see the top-level README). Remaining open problem for this path: replicate llama.cpp's image preprocessing to close the image gap.

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

Verify the pooling recipe against the GGUF path (two steps, see the file's docstring):

```bash
# with ./embedder.sh start running, in the turbovec venv:
python match_recipe.py capture
# then stop the embedder, free VRAM, in the transformers venv:
python match_recipe.py match     # expect "pre last" ~0.9985
```

## Files

| file | purpose |
|---|---|
| `embed_tf.py` | transformers embedder: text / image / audio / native video |
| `compare.py` | native video vs single-frame vs audio-only, for a given clip |
| `match_recipe.py` | reverse-engineer LCO's pooling by matching the GGUF output |
| `download_hf_model.sh` | fetch the non-GGUF weights |
| `requirements-transformers.txt` | deps (install torch separately) |
