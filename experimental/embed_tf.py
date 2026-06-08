#!/usr/bin/env python3
"""
LCO-Embedding-Omni via HF transformers (non-GGUF) - the full-fidelity path.

Unlike the GGUF/llama.cpp path, this can do NATIVE multi-frame video (frames +
audio fused into one vector). It is the reference/research path: heavier, and in
our testing its retrieval quality is below the GGUF path (see experimental/README.md).

Embedding = last token of the final decoder layer's PRE-final-norm hidden state
(`hidden_states[-1]`), L2-normalized. This was reverse-engineered by matching the
GGUF/llama.cpp `--pooling last` output: it reproduces the GGUF vectors at mean
cosine 0.9985 (vs 0.96 for post-norm last, 0.52 for mean pooling). See
match_recipe.py.

Env:
  LCO_HF_MODEL   path to the non-GGUF model dir (default ~/models/lco-omni-hf)
"""
import os
import numpy as np
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MODEL = os.path.expanduser(os.environ.get("LCO_HF_MODEL", "~/models/lco-omni-hf"))
_model = _proc = None

def load():
    global _model, _proc
    if _model is None:
        _proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
        _model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    return _model, _proc

def _np(v):
    v = v.float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)

def _pool_last(out):
    # LCO recipe: last token of the final layer's PRE-norm hidden state.
    return _np(out.hidden_states[-1][0, -1])

@torch.no_grad()
def embed_text(t):
    model, proc = load()
    enc = proc.tokenizer(t, return_tensors="pt").to(model.device)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                output_hidden_states=True, return_dict=True)
    return _pool_last(out)

@torch.no_grad()
def _embed_media(content, use_audio_in_video=False):
    model, proc = load()
    conv = [{"role": "user", "content": content}]
    audios, images, videos = process_mm_info(conv, use_audio_in_video=use_audio_in_video)
    # bare media markers (no chat scaffold) so the LAST token is media content,
    # matching the GGUF path's prompt_string="<__media__>" construction.
    parts = []
    for c in content:
        if c["type"] == "image": parts.append(f"<|vision_start|>{proc.image_token}<|vision_end|>")
        elif c["type"] == "video": parts.append(f"<|vision_start|>{proc.video_token}<|vision_end|>")
        elif c["type"] == "audio": parts.append(f"<|audio_bos|>{proc.audio_token}<|audio_eos|>")
    text = "".join(parts)
    inputs = proc(text=text, audio=audios, images=images, videos=videos,
                  return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video).to(model.device)
    out = model(**inputs, output_hidden_states=True, return_dict=True, use_audio_in_video=use_audio_in_video)
    return _pool_last(out)

def embed_image(path):
    return _embed_media([{"type": "image", "image": path}])

def embed_audio(path):
    return _embed_media([{"type": "audio", "audio": path}])

def embed_video(path, with_audio=True, fps=1.0, max_pixels=None):
    item = {"type": "video", "video": path, "fps": fps}
    if max_pixels:
        item["max_pixels"] = max_pixels
    return _embed_media([item], use_audio_in_video=with_audio)

if __name__ == "__main__":
    a, b, c = embed_text("a fluffy cat"), embed_text("a domestic kitten"), embed_text("a financial report")
    print(f"dim={a.shape[0]}  cos(cat,kitten)={float(a@b):+.3f}  cos(cat,finance)={float(a@c):+.3f}")
