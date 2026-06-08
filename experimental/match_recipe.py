#!/usr/bin/env python3
"""
Reverse-engineer LCO's embedding recipe by matching the GGUF/llama.cpp output.

The GGUF path (`--pooling last`) is the validated reference. This finds the
transformers-side extraction that reproduces its vectors. Result: the embedding
is the LAST token of the final decoder layer's PRE-final-norm hidden state
(`hidden_states[-1]`), L2-normalized - mean cosine 0.9985 to GGUF on text
(vs 0.96 post-norm-last, 0.52 mean-pool). That recipe is what embed_tf.py uses.

Two steps (the GGUF embedder and the bf16 model don't fit in VRAM together):

  # 1. with the GGUF embedder running (./embedder.sh start), in the turbovec venv:
  python match_recipe.py capture          # saves gguf_vecs.npy

  # 2. stop the embedder, free VRAM, then in the transformers venv:
  python match_recipe.py match            # loads gguf_vecs.npy, sweeps variants

Env: LCO_SERVER (capture), LCO_HF_MODEL (match).
"""
import os, sys, numpy as np

TEXTS = [
    "A fluffy domestic cat grooming itself.", "A small kitten playing with yarn.",
    "A loyal dog playing fetch in the park.", "The Eiffel Tower in Paris at sunset.",
    "Quarterly financial earnings report.", "A recipe for tomato marinara sauce.",
    "Ocean waves crashing on a sandy beach.", "A train passing through a railway station.",
    "Computer code running in a terminal.", "A child riding a bicycle.",
    "The history of the Roman Empire.", "A cup of hot coffee on a wooden table.",
]
VECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_vecs.npy")

def capture():
    import requests
    S = os.environ.get("LCO_SERVER", "http://127.0.0.1:8090")
    def emb(t):
        r = requests.post(f"{S}/embedding", json={"content": t}, timeout=60); r.raise_for_status()
        j = r.json(); e = j[0]["embedding"] if isinstance(j, list) else j["embedding"]
        return np.asarray(e, dtype=np.float32).reshape(-1)
    np.save(VECS, np.vstack([emb(t) for t in TEXTS]))
    print(f"saved {VECS}")

def match():
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    MODEL = os.path.expanduser(os.environ.get("LCO_HF_MODEL", "~/models/lco-omni-hf"))
    G = np.load(VECS); Gu = G / np.linalg.norm(G, axis=1, keepdims=True)
    proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    def unit(v): v = v.float().cpu().numpy(); return v / (np.linalg.norm(v) + 1e-9)

    @torch.no_grad()
    def hs(text):
        enc = proc.tokenizer(text, return_tensors="pt").to(model.device)
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    output_hidden_states=True, return_dict=True)
        pre = out.hidden_states[-1][0]
        post = model.model.norm(out.hidden_states[-1])[0]
        return pre, post

    cache = [hs(t) for t in TEXTS]
    variants = {
        "pre  last": lambda p, q: unit(p[-1]),
        "post last": lambda p, q: unit(q[-1]),
        "pre  mean": lambda p, q: unit(p.mean(0)),
        "post mean": lambda p, q: unit(q.mean(0)),
    }
    print("match to GGUF (mean cosine; 1.0 = exact recipe):")
    for name, fn in variants.items():
        vs = np.vstack([fn(p, q) for p, q in cache])
        cos = float(np.mean([vs[i] @ Gu[i] for i in range(len(TEXTS))]))
        print(f"  {cos:+.4f}   {name}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "capture": capture()
    elif cmd == "match": match()
    else: sys.exit("usage: match_recipe.py capture|match")
