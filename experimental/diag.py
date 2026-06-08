#!/usr/bin/env python3
"""
Diagnostic used to choose the embedding extraction for the transformers path.

LLM hidden states are anisotropic, so HOW you pool matters a lot. This compares
last-token vs mean pooling on post-final-norm hidden states and prints pairwise
cosines on a tiny semantic probe set. Mean-pool gave the cleanest ordering
(similar pairs clearly above dissimilar ones), which is what embed_tf.py uses.

Env: LCO_HF_MODEL (default ~/models/lco-omni-hf)
"""
import os
import numpy as np
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

MODEL = os.path.expanduser(os.environ.get("LCO_HF_MODEL", "~/models/lco-omni-hf"))
proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
tok = proc.tokenizer
base = model.model   # last_hidden_state is POST final-norm

texts = {"cat": "A fluffy domestic cat grooming itself.",
         "kitten": "A small kitten playing.",
         "finance": "Quarterly financial earnings report.",
         "eiffel": "The Eiffel Tower in Paris."}

def norm(v):
    v = v.float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)

@torch.no_grad()
def lhs(t):
    enc = tok(t, return_tensors="pt").to(model.device)
    out = base(input_ids=enc.input_ids, attention_mask=enc.attention_mask, return_dict=True)
    return out.last_hidden_state[0]

def report(name, fn):
    e = {k: fn(v) for k, v in texts.items()}
    s = lambda a, b: float(e[a] @ e[b])
    print(f"{name:14} cat/kitten={s('cat','kitten'):+.3f}  cat/finance={s('cat','finance'):+.3f}  "
          f"cat/eiffel={s('cat','eiffel'):+.3f}  finance/eiffel={s('finance','eiffel'):+.3f}")

if __name__ == "__main__":
    report("postnorm_last", lambda t: norm(lhs(t)[-1]))
    report("postnorm_mean", lambda t: norm(lhs(t).mean(0)))
