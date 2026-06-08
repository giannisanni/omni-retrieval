# omni-retrieval

**Fully-local, air-gapped cross-modal retrieval across text, images, audio, and video.** One model embeds everything into a single shared vector space (video by fusing sampled frames and its audio), and a compact 4-bit index runs fast nearest-neighbor search on CPU. Type a plain-language description and get back the matching document, photo, audio clip, or video. It matches on content rather than metadata, so your files need no captions or tags, and it all runs on your own hardware.

It pairs:

- **[LCO-Embedding-Omni-3B-2605](https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-3B-2605)** - a contrastively fine-tuned Qwen2.5-Omni-Thinker that emits **2048-dim** embeddings for text / image / audio / video into one shared space, served locally via `llama.cpp`.
- **[turbovec](https://github.com/RyanCodrai/turbovec)** - a CPU-resident `IdMapIndex` using 4-bit TurboQuant (~512 bytes/vector, ~5 GB per 10M items), with stable uint64 ids and O(1) deletes.

The result is a `lcovec` CLI: `ingest` a folder, `query` it in natural language.

```text
$ lcovec.py query "a kitten"

query: 'a kitten'

  score=+1.83 (z=+1.16 cos=+0.224)  [text ] A fluffy domestic cat grooming itself by the window.
  score=+1.46 (z=+1.12 cos=+0.111)  [image] cat.jpg
  score=+1.38 (z=+1.06 cos=+0.108)  [image] dog.jpg
  score=+0.97 (z=+0.49 cos=+0.161)  [text ] A loyal dog playing fetch at the park.
```

A text query retrieving an *image* by its visual content, with no metadata on the file, is the point.

---

## Two ways to run it

| | Path A: lite (recommended) | Path B: full (experimental) |
|---|---|---|
| backend | GGUF + llama.cpp | HF transformers |
| this is | the `lcovec` CLI (this README) | [`experimental/`](experimental/) |
| footprint | ~9 GB VRAM, fast, CPU index | full bf16 model, heavier |
| video | multi-frame + audio fused (one vector) | native temporal frames + audio |
| retrieval quality | clean (4/4 on the cross-modal probe) | text-exact (0.9985 to GGUF), images 3/4 (preprocessing gap) |
| use it for | actually using/searching, and demos | research, native-video work, development |

**Most people want Path A** (the rest of this README). Path B exists for native video and capability research; it is honest about currently underperforming Path A for retrieval. See [experimental/README.md](experimental/README.md).

---

## Contents

- [How it works](#how-it-works)
- [Two gotchas (read this first)](#two-gotchas-read-this-first)
- [Requirements](#requirements)
- [Install](#install)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [The modality gap and how it is fixed](#the-modality-gap-and-how-it-is-fixed)
- [What actually works (measured)](#what-actually-works-measured)
- [Performance and footprint](#performance-and-footprint)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Credits](#credits)
- [License](#license)

---

## How it works

```mermaid
flowchart LR
    subgraph SRC["mixed-media corpus"]
        T["text"]
        I["images"]
        A["audio"]
        V["video"]
    end
    subgraph GPU["embedder (GPU, ~9 GB)"]
        E["LCO-Embedding-Omni-3B / llama.cpp fork / --embedding --pooling last"]
    end
    subgraph CPU["index (CPU + RAM)"]
        X["turbovec IdMapIndex / dim 2048, 4-bit / stable uint64 ids"]
    end
    Q["text query"]
    R["ranked results"]
    SRC -->|ingest| E
    E -->|2048-d vectors| X
    Q --> E
    E -.->|query vector| X
    X -.->|top-k ids + scores| R
```

The pipeline has four stages.

**1. Embedding into one shared space.** A single model, LCO-Embedding-Omni-3B, turns any input (a sentence, a photo, an audio clip) into one 2048-dimensional vector. It was contrastively trained to place semantically related inputs near each other *regardless of modality*, so the sentence "a sleeping cat" and a photograph of a sleeping cat land close together in the same space. That shared geometry is what makes typing words and getting back a matching image possible. The model is served by a `llama.cpp` fork with `--pooling last` (the embedding is the final token's hidden state) and the vectors come out L2-normalized, so similarity is a plain dot product (cosine).

**2. Ingestion, routed by file type.** `ingest` walks the paths you give it and handles each file according to its extension:

- **text** (`.txt`, `.md`, ...): the file's contents are read and embedded directly.
- **images** and **audio** (`.jpg`, `.wav`, ...): the raw file is base64-encoded and embedded as media.
- **PDF** (`.pdf`): one item **per page** (PyMuPDF). Pages with extractable text are embedded as text (the strongest mode); scanned/image-only pages are rendered to an image and embedded as such. A query lands you on the matching page, not just the file.
- **video** (`.mp4`, ...): llama.cpp's multimodal stack does not decode video containers, so a raw `.mp4` is rejected. `lcovec` does the decode itself: it uses `ffmpeg` to sample several frames spread across the clip (default 6) plus the audio track, and sends them as **one fused multimodal request** so the model returns a single multi-frame + audio **video** vector. Tune with `LCOVEC_VIDEO_FRAMES` / `LCOVEC_VIDEO_AUDIO_SEC`. This captures the whole clip, not just one frame; the remaining gap vs a true native pipeline is temporal frame-merging (see [Native video](#native-video)). The audio component is subject to the speech caveat below.

Re-running `ingest` skips files already in the index.

**3. Indexing.** Each vector is stored in a turbovec `IdMapIndex` under a stable uint64 id, while the file's path and modality are kept in a small JSON sidecar. turbovec quantizes every vector to 4 bits per dimension (TurboQuant), which is what lets ~10M items fit in roughly 5 GB of RAM and keeps nearest-neighbor search fast on CPU, with no GPU needed once the corpus is embedded. Ids survive deletion, so a corpus that constantly gains and loses files stays consistent without rebuilds.

**4. Querying, with cross-modal calibration.** Your text query is embedded by the same model, then searched against the index. A naive nearest-neighbor lookup is biased: text-to-text similarities run systematically higher than text-to-image, text-to-audio, or text-to-video, so correct non-text items sink beneath unrelated text. `lcovec` corrects this by calibrating **each item** against a fixed bank of probe queries (computed once at ingest), standardizing each result against its own baseline, and blending with the raw cosine. This works even for a modality with a single item, so a lone video is still findable. The mechanism is detailed in [The modality gap and how it is fixed](#the-modality-gap-and-how-it-is-fixed).

---

## Two gotchas (read this first)

These two settings are the difference between working embeddings and silent garbage. Both are handled for you by `embedder.sh`, but if you roll your own launch command you **will** hit them. Symptom of either: every image/audio item returns a near-identical vector (cosine ~1.0 between unrelated files).

1. **The fork randomizes the media placeholder.** For prompt-injection hardening, the server's media marker is `<__media_<random>__>` per process unless you pin it. If you send the literal `<__media__>` without pinning, it is tokenized as plain text, the media is dropped, and every item collapses to the same vector. **Fix:** launch with `LLAMA_MEDIA_MARKER='<__media__>'`.

2. **The CLIP graph uses CUDA ops the backend rejects.** Warmup prints `the CLIP graph uses unsupported operators by the backend` and the vision/audio projector produces degenerate output. **Fix:** run the projector on CPU with `--no-mmproj-offload`.

And the request shape: media goes **inside** `content`, not at the top level as the model card implies:

```json
{"content": {"prompt_string": "<__media__>", "multimodal_data": ["<base64>"]}}
```

Plain text stays simple: `{"content": "your text"}`. The endpoint is `POST /embedding`.

---

## Requirements

- A **CUDA GPU with ~9 GB free VRAM** (built/tested on an RTX 4060 Ti 16 GB). The projector runs on CPU, so CPU-only inference is possible but slow.
- `git`, `cmake`, a CUDA toolkit (to build the fork).
- `python3` with `pip`.
- `ffmpeg` (only for audio/video ingestion); `pymupdf` (only for PDF, installed via requirements).
- ~6 GB disk for the model weights.

This loads the multimodal **embedding** path of Qwen2.5-Omni, which mainline `llama.cpp` and Ollama do not ship. You must build the **[ht-llama.cpp](https://github.com/heiervang-technologies/ht-llama.cpp)** fork.

---

## Install

```bash
git clone https://github.com/giannisanni/omni-retrieval
cd omni-retrieval
pip install -r requirements.txt        # turbovec, numpy, requests

# 1. build the llama.cpp fork (set CMAKE_CUDA_ARCHITECTURES for your GPU:
#    89 = Ada/RTX 40xx, 86 = Ampere, 75 = Turing)
CMAKE_CUDA_ARCHITECTURES=89 ./scripts/build_llama_fork.sh

# 2. download the LCO-Omni GGUF weights + mmproj (~6 GB)
./scripts/download_model.sh
```

By default the fork lands in `~/ht-llama.cpp` and the model in `~/models/lco-omni`. Override with the script arguments or the env vars listed in [Configuration](#configuration).

---

## Quick start

```bash
# start the embedder (load-on-demand; needs ~9 GB VRAM)
./embedder.sh start

# index a folder of mixed media (text, images, audio, video)
./lcovec.py ingest ~/some/folder

# search it in natural language
./lcovec.py query "a red bicycle leaning against a wall" -k 8
./lcovec.py stats

# stop it again to free the GPU
./embedder.sh stop
```

Want the canned demo from the top of this README?

```bash
./scripts/fetch_sample_images.sh     # 4 CC images into ./images/
./embedder.sh start
./poc.py                             # runs the cross-modal proof end to end
```

---

## CLI reference

| command | description |
|---|---|
| `lcovec.py ingest <path>...` | Walk files/dirs, embed by type, add to the index. Re-ingesting skips known sources. |
| `lcovec.py query "<text>" [-k N]` | Cross-modal search. `-k` is the number of results (default 6). |
| `lcovec.py stats` | Item counts per modality. |
| `lcovec.py reset` | Wipe the index and metadata. |
| `embedder.sh start \| stop \| status` | Manage the embedding server. |

Recognized extensions: text (`.txt .md .markdown .rst`), image (`.jpg .jpeg .png .gif .bmp .webp`), audio (`.mp3 .wav .m4a .aac .flac .ogg`), video (`.mp4 .mov .mkv .webm .avi`), pdf (`.pdf`, page by page).

---

## The modality gap and how it is fixed

Embeddings cluster by modality: text-to-text cosines run systematically higher than text-to-image, text-to-audio, or text-to-video, even when the cross-modal match is correct. In a naive single ranked list this buries correct non-text items beneath unrelated text.

`lcovec` calibrates **each item** against a fixed bank of ~16 diverse probe queries:

1. **At ingest**, compute each item's baseline: the mean and std of its cosine to the probe bank, stored with the item. (The probe embeddings are cached once.)
2. **At query**, standardize against that baseline: `z = (cos - mean) / std` - how unusual this match is *for that specific item*.
3. **Blend** lightly with raw cosine so strong absolute matches (text) stay on top and irrelevant low-baseline items do not float up:

   ```text
   score = z + cos
   ```

Per-*item* (not just per-modality) calibration matters because anisotropy varies *within* a modality. For example a speech-bearing video is a "text magnet" - it has high cosine to almost every text query; subtracting its own baseline cancels that. This also works for a modality with a single item, where result-set z-scoring fails outright. In testing it lifts a correct image to the top of a mixed list and takes video-only retrieval from 2/5 to 4/5 (see below).

---

## What actually works (measured)

Small controlled probes on the reference hardware. These are descriptive, not a benchmark.

**Text -> image** (image-only corpus, 4 distinct images): **4/4** queries rank the correct image first. The image space is semantically structured (cat/dog pairwise cosine 0.52, the highest pair; unrelated pairs 0.32-0.41).

**Text -> audio depends on whether the audio contains speech.**

| clip type | best text->audio cosine | retrievable by text? |
|---|---|---|
| speech (spoken monologue) | 0.097, all content-matching queries beat all distractors (mean 0.084 vs 0.032) | **yes** |
| sound-effects only (no speech) | 0.038, no separation from distractors | no |

The model is language-centric: it aligns audio to words well when the audio carries **speech**, and poorly for pure sound effects or music. Retrieve non-speech audio by **audio-to-audio** similarity instead of text. Absolute audio cosines stay low (~0.07-0.10 even for matches), which is exactly why the per-item standardization above matters.

**Text -> video** (5 content-distinct clips: cat, train, ocean, cooking, and a spoken tech monologue; `.mp4` decoded to 6 frames + audio and fused). Among videos only, retrieving the correct clip by a content query:

| ranking | accuracy |
|---|---|
| raw cosine | 2/5 |
| per-item probe-calibrated | **4/5** |

Raw cosine fails because the spoken-monologue clip is a "text magnet" (high cosine to every query); per-item calibration cancels that. **In a mixed corpus (videos + text + images), video retrieval is weak**: visual-dominated clips align loosely and noisily to text, so they are out-competed by text/image and do not reliably reach the top results. Only the speech-bearing clip surfaces near the top for its query. Treat fused video as best-effort; text -> image and text -> speech-audio are the dependable cross-modal cases.

| capability | status |
|---|---|
| text -> text | strong |
| text -> image | strong |
| text -> audio (speech) | works (rank-correct, low absolute scores) |
| text -> audio (non-speech) | unreliable; use audio-to-audio |
| text -> video (among videos) | 4/5 with per-item calibration |
| text -> video (mixed corpus) | weak/noisy for visual clips; speech clips surface |
| text -> PDF (per page) | strong (text pages use the text mode); lands on the matching page |
| add / search / delete with stable ids | works (O(1) delete) |

---

## Benchmark

A larger content-distinct corpus: 8 text/markdown docs, 7 images, 4 arXiv PDFs (68 pages), 3 speech clips, 4 videos = **90 indexed items**, with 26 labeled queries (one per source item; a PDF counts as found if any of its pages ranks). Scoring is the shipped per-item-calibrated `z + cos`. Reproduce with [`benchmark/`](benchmark/) (it fetches the corpus fresh, so numbers vary slightly).

**Within-modality** (find the right item among items of the same type):

| modality | top-1 |
|---|---|
| text | 6/8 |
| image | 6/7 |
| pdf (page) | 3/4 |
| audio (speech) | 2/3 |
| video | 3/4 |
| **overall** | **20/26 (77%)** |

**Mixed corpus** (all 90 items compete for every query):

| metric | value |
|---|---|
| top-1 | 16/26 (62%) |
| top-3 | 20/26 (77%) |
| MRR | 0.71 |

Two honest reads of the misses:

- **Most are semantically reasonable, not failures.** The machine-learning text loses to the ML *PDFs*; the Eiffel Tower *image* loses to Paris/France *text*; cat vs dog; the transformer papers (attention/BERT) shade into each other. The embeddings are behaving sensibly.
- **Audio and video get buried in a text-heavy mixed corpus.** Within their own modality they retrieve fine (audio 2/3, video 3/4), but in the mixed index they sink to ranks ~15-80 of 90 - text->audio/video cosines are low and 76 of 90 items are text. Per-item calibration narrows this but cannot overcome a ~25:1 modality imbalance. If you need every modality represented in one list, search per-modality and merge top-k from each.

Bottom line, at scale: **text, image, and PDF retrieval are strong; speech-audio works; cross-modal audio/video ranking against a large text corpus is the weak spot.**

---

## Native video

The LCO model architecture (Qwen2.5-Omni) genuinely supports video. The limitation is the **serving layer**, not the model. In the reference pipeline (HF transformers / vLLM), a processor decodes the container, samples frames over time, and extracts audio before the model ever sees it, so "native video" is really frames + audio assembled by the processor.

This project serves the GGUF through **llama.cpp's multimodal stack (mtmd)**, which accepts already-decoded image and audio bitmaps but has no video-container decode at the `/embedding` endpoint. A raw `.mp4` returns `HTTP 500 "Failed to load image or audio file"`. `lcovec` does the decode itself with `ffmpeg`: it samples several frames across the clip plus the audio and sends them as one fused multimodal request, so the GGUF path produces a real **multi-frame + audio** video vector. This is the practical fix and keeps the lightweight footprint.

What it is *not* yet: true temporal video. mtmd merges at most two consecutive frames (`n_merge_frames <= 2`), so the frames are largely treated as a set of images rather than a time-ordered sequence with full temporal position encoding. For that, serve LCO through transformers or vLLM with the Qwen2.5-Omni processor; a working transformers implementation lives in [`experimental/`](experimental/) (Path B).

Empirical caveats from a 5-clip benchmark: among videos, per-item calibration retrieves the right clip 4/5 (raw cosine only 2/5, because a spoken-monologue clip is a text magnet). But in a **mixed** corpus, visual-dominated clips align loosely/noisily to text and are out-competed by text and images; only speech-bearing clips reliably surface. Fused video also dilutes audio when many frames are used. Native fused video helps most for visually-distinct content, not as a drop-in equal of text/image retrieval. Details in [experimental/README.md](experimental/README.md).

---

## Performance and footprint

- **VRAM:** ~9 GB resident (Q8 weights 3.4 GB + F16 mmproj 2.5 GB + KV cache + CUDA context). Fits a 16 GB card with room to spare.
- **Index:** ~512 bytes per vector at 4-bit (~5 GB for 10M items), held in CPU RAM. 2-bit halves it.
- **Throughput:** text embeds are fast; media embeds run the projector on CPU (~30 s per ~15 s audio clip on an 8-core box). Fine for batch ingestion, not for high-QPS media ingest. Per-item calibration is computed at ingest (the ~16 probe queries are embedded once), so querying adds no probe overhead.
- **Context:** 8192 tokens (`embedder.sh`). Audio is ~100 tokens/second; fused video uses `LCOVEC_VIDEO_FRAMES` frames (~256 tokens each) plus `LCOVEC_VIDEO_AUDIO_SEC` of audio, kept within this budget.

---

## When do I need to re-embed?

You embed each item **once** and reuse the index indefinitely. The index is persisted to `~/.lcovec/store`, so it survives restarts, and search runs on CPU with the embedder unloaded.

- **Adding files:** no re-embed of existing data. `ingest` only embeds files not already in the index.
- **Querying, restarting, moving the store:** no re-embed. Just point `LCOVEC_STORE` at it.
- **Changing the embedding model:** **full re-embed required.** Embeddings are model-specific. A vector from one model and a vector from another live in different geometric spaces and are not comparable, even at the same dimensionality, so you cannot search across them or mix them in one index. Switching models (or even a different quantization that changes the model's output) means rebuilding the index from scratch.

In other words: the index is decoupled from the embedder and will store whatever vectors you give it, but the vectors themselves are tied to the model that produced them. This tool is built specifically around LCO-Embedding-Omni (the dimension and multimodal request format are fixed to it); pointing it at a different model takes code changes and a fresh index.

---

## Configuration

| variable | default | used by |
|---|---|---|
| `LCO_SERVER` | `http://127.0.0.1:8090` | `lcovec.py`, `poc.py` |
| `LCOVEC_STORE` | `~/.lcovec/store` | `lcovec.py` |
| `LCOVEC_VIDEO_FRAMES` | `6` | `lcovec.py` (frames sampled per video) |
| `LCOVEC_VIDEO_AUDIO_SEC` | `45` | `lcovec.py` (seconds of audio fused per video) |
| `LLAMA_SERVER_BIN` | `~/ht-llama.cpp/build/bin/llama-server` | `embedder.sh` |
| `LCO_MODEL` | `~/models/lco-omni/LCO-Embedding-Omni-3B-2605-Q8_0.gguf` | `embedder.sh` |
| `LCO_MMPROJ` | `~/models/lco-omni/mmproj-LCO-Embedding-Omni-3B-2605-F16.gguf` | `embedder.sh` |
| `LCO_PORT` | `8090` | `embedder.sh` |

---

## Project layout

```text
omni-retrieval/
  lcovec.py            # the CLI: ingest / query / stats / reset
  embedder.sh          # start/stop the embedding server (pins the two gotchas)
  poc.py               # end-to-end cross-modal proof on sample data
  requirements.txt
  scripts/
    build_llama_fork.sh    # clone + CUDA-build the ht-llama.cpp fork
    download_model.sh      # fetch the LCO-Omni GGUF + mmproj
    fetch_sample_images.sh # 4 CC images for poc.py
  experimental/          # Path B: transformers / native-video (research, see its README)
    embed_tf.py  compare.py  match_recipe.py  download_hf_model.sh
  benchmark/             # reproducible multimodal benchmark (download_corpus.py, run.py)
```

Index and metadata persist under `~/.lcovec/store` (`index.tvim`, `meta.json`, `derived/`).

---

## Limitations

This is an engineering proof-of-concept, validated but not a benchmark study. The probe corpora are small, so reported numbers are descriptive with no confidence intervals. Cross-modal accuracy was measured on a semantically disjoint set and will overstate performance on fine-grained or near-duplicate corpora. Absolute cosine magnitudes are not comparable across modalities by construction (the modality gap), so ranking, not raw score, is the operative quantity. There is no folder-watcher, web UI, or auth; it is a CLI over a local index.

---

## Contributing

Issues and PRs welcome. Please run `ruff check .` and `shellcheck embedder.sh scripts/*.sh` first (CI runs both). See [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

- **LCO-Embedding-Omni** - [model](https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-3B-2605), [GGUF build](https://huggingface.co/marksverdhei/LCO-Embedding-Omni-3B-2605-GGUF), paper [arXiv:2510.11693](https://arxiv.org/abs/2510.11693). Built on [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-3B).
- **turbovec** - [repo](https://github.com/RyanCodrai/turbovec), TurboQuant paper [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
- **ht-llama.cpp** - [fork](https://github.com/heiervang-technologies/ht-llama.cpp) adding the Qwen2.5-Omni embedding arch, on top of [llama.cpp](https://github.com/ggml-org/llama.cpp).

This repo's own code is Apache-2.0; the model and dependencies carry their own licenses.

## License

[Apache License 2.0](LICENSE).
