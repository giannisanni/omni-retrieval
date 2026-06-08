#!/usr/bin/env python3
"""
Multimodal retrieval benchmark. Reports per-modality top-1 and mixed-corpus
top-1/top-3/MRR using the shipped per-item-calibrated scoring.

Usage (after building the corpus with download_corpus.py and starting the embedder):
  LCOVEC_STORE=~/bench/store python ../lcovec.py reset
  LCOVEC_STORE=~/bench/store python ../lcovec.py ingest ~/bench/corpus
  LCOVEC_STORE=~/bench/store python run.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lcovec

idx, meta = lcovec.load_store()
items = {int(k): v for k, v in meta["items"].items()}
by = lcovec._by_mod(meta)
MOD_MAP = {"text": ["text"], "image": ["image"], "audio": ["audio"],
           "video": ["video"], "pdf": ["text", "image"]}

# (manifest modality, filename key, query). A PDF is "found" if any of its pages ranks.
M = [
 ("text", "Photosynthesis", "how plants convert sunlight into chemical energy"),
 ("text", "French_Revolution", "the overthrow of the French monarchy starting in 1789"),
 ("text", "Black_hole", "a region of spacetime where gravity stops light escaping"),
 ("text", "Espresso", "concentrated coffee forced through finely ground beans"),
 ("text", "Basketball", "a team sport shooting a ball through a hoop"),
 ("text", "Roman_Empire", "the ancient empire ruled from Rome by emperors"),
 ("text", "Machine_learning", "algorithms that learn patterns from data"),
 ("text", "Volcano", "a mountain that erupts molten lava and ash"),
 ("image", "cat", "a photo of a cat"),
 ("image", "dog", "a photo of a dog"),
 ("image", "eiffel", "the Eiffel Tower in Paris"),
 ("image", "car", "a car parked on a street"),
 ("image", "Airplane", "a commercial airplane in flight"),
 ("image", "Mountain", "a snowy mountain landscape"),
 ("image", "Sailing_ship", "a sailing ship on the sea"),
 ("pdf", "attention", "the transformer self-attention mechanism for sequences"),
 ("pdf", "resnet", "deep residual learning for image recognition"),
 ("pdf", "bert", "bidirectional transformer pretraining for language understanding"),
 ("pdf", "turboquant", "data-oblivious quantization for nearest neighbor search"),
 ("audio", "coffee", "how espresso coffee is made"),
 ("audio", "blackhole", "an explanation of black holes"),
 ("audio", "basketball", "the rules of the game of basketball"),
 ("video", "cat", "a cat"),
 ("video", "train", "a train on railway tracks"),
 ("video", "cooking", "frying food in a pan"),
 ("video", "guitar", "someone playing a guitar"),
]

# topic of each item/query (for the fair "did we surface the right topic, in any
# modality" metric - the corpus deliberately has cross-modal topic duplicates).
KEY2TOPIC = {"Espresso": "coffee", "coffee": "coffee", "Black_hole": "blackhole",
    "blackhole": "blackhole", "Basketball": "basketball", "basketball": "basketball",
    "cat": "cat", "dog": "dog", "eiffel": "eiffel", "Eiffel": "eiffel", "car": "car",
    "Airplane": "airplane", "Mountain": "mountain", "Sailing_ship": "ship", "train": "train",
    "cooking": "cooking", "guitar": "guitar", "Photosynthesis": "photosynthesis",
    "French_Revolution": "french", "Roman_Empire": "roman", "Machine_learning": "ml",
    "Volcano": "volcano", "attention": "attention", "resnet": "resnet", "bert": "bert",
    "turboquant": "turboquant"}
def topic_of(s):
    for k, t in KEY2TOPIC.items():
        if k in s: return t
    return None
itopic = {i: topic_of(os.path.basename(v["path"])) for i, v in items.items()}

def targets(key):
    return [i for i, v in items.items() if key in os.path.basename(v["path"])]

def scored(q, mods=None):
    qv = lcovec.embed_text(q).reshape(1, -1).astype(np.float32)
    out = []
    for m in (mods or list(by)):
        if m not in by: continue
        sc, rid = idx.search(qv, k=len(by[m]), allowlist=np.array(by[m], dtype=np.uint64))
        for s, r in zip(sc[0], rid[0]):
            it = items[int(r)]
            out.append(((float(s) - it.get("bm", 0.0)) / it.get("bs", 1.0) + lcovec.BLEND * float(s), int(r)))
    out.sort(reverse=True)
    return [r for _, r in out]

def rank_of(order, pred):
    for pos, r in enumerate(order, 1):
        if pred(r): return pos
    return 0

per = {}; t1 = t3 = 0; rr = 0.0; gt1 = gt3 = 0; grr = 0.0; n = 0; missing = []
print(f"BLEND={lcovec.BLEND}\n{'modality':7} {'target':16} modal# mixed#")
for modality, key, q in M:
    tgt = set(targets(key)); qt = topic_of(key)
    if not tgt: missing.append(f"{modality}:{key}"); continue
    n += 1
    order = scored(q)
    mr = rank_of(scored(q, MOD_MAP[modality]), lambda r: r in tgt)
    xr = rank_of(order, lambda r: r in tgt)              # strict: the labeled item
    gr = rank_of(order, lambda r: itopic[r] == qt)       # graded: right topic, any modality
    d = per.setdefault(modality, [0, 0, 0]); d[0] += 1; d[1] += (mr == 1); d[2] += (xr == 1)
    t1 += (xr == 1); t3 += (0 < xr <= 3); rr += (1.0/xr if xr else 0)
    gt1 += (gr == 1); gt3 += (0 < gr <= 3); grr += (1.0/gr if gr else 0)
    print(f"{modality:7} {key:16} {mr:>5} {xr:>6}")
print("\nper-modality top-1 (within | mixed):")
for m, (tot, a, b) in per.items(): print(f"  {m:6}: within {a}/{tot}   mixed {b}/{tot}")
print(f"\nstrict (exact labeled item): top-1 {t1}/{n} ({100*t1/n:.0f}%)  top-3 {t3}/{n} ({100*t3/n:.0f}%)  MRR {rr/n:.3f}")
print(f"topic-graded (right topic, any modality): top-1 {gt1}/{n} ({100*gt1/n:.0f}%)  top-3 {gt3}/{n} ({100*gt3/n:.0f}%)  MRR {grr/n:.3f}")
if missing: print("missing:", missing)
