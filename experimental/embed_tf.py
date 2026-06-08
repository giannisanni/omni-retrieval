#!/usr/bin/env python3
"""
LCO-Embedding-Omni via HF transformers (non-GGUF) - the full-fidelity path.

Unlike the GGUF/llama.cpp path, this can do NATIVE multi-frame video (frames +
audio fused into one vector). It is the reference/research path: heavier, and in
our testing its retrieval quality is below the GGUF path (see experimental/README.md).

Embedding = mean-pool over the Thinker's post-final-norm hidden states,
L2-normalized. This was the extraction with the cleanest semantic ordering we
found (see diag.py); it is an approximation of LCO's exact recipe, which the
llama.cpp path encodes and the model repo does not document.

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

def _meanpool(h, mask):           # h:(1,seq,2048), mask:(1,seq)
    m = mask.unsqueeze(-1).to(h.dtype)
    return ((h * m).sum(1) / m.sum(1).clamp(min=1))[0]

@torch.no_grad()
def embed_text(t):
    model, proc = load()
    enc = proc.tokenizer(t, return_tensors="pt").to(model.device)
    out = model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, return_dict=True)
    return _np(_meanpool(out.last_hidden_state, enc.attention_mask))   # last_hidden_state is post-norm

@torch.no_grad()
def _embed_media(content, use_audio_in_video=False):
    model, proc = load()
    conv = [{"role": "system", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": content}]
    text = proc.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=use_audio_in_video)
    inputs = proc(text=text, audio=audios, images=images, videos=videos,
                  return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video).to(model.device)
    out = model(**inputs, output_hidden_states=True, return_dict=True, use_audio_in_video=use_audio_in_video)
    h = model.model.norm(out.hidden_states[-1])    # apply final RMSNorm -> post-norm
    return _np(_meanpool(h, inputs.attention_mask))

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
