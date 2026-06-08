#!/usr/bin/env python3
"""
turbovec + LCO-Embedding-Omni cross-modal retrieval PoC.

ONE model embeds text AND images into ONE 2048-d space; turbovec IdMapIndex
(4-bit TurboQuant, CPU) indexes them so a TEXT query retrieves the matching
IMAGE. Requires the ht-llama.cpp fork serving with:
  LLAMA_MEDIA_MARKER='<__media__>' llama-server -m <q8> --mmproj <f16> \
     --no-mmproj-offload --embedding --pooling last
"""
import os, base64, numpy as np, requests
from turbovec import IdMapIndex

SERVER = os.environ.get("LCO_SERVER", "http://127.0.0.1:8090")
IMG    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
DIM    = 2048

def _vec(resp):
    e = resp[0]["embedding"] if isinstance(resp, list) else resp["embedding"]
    a = np.asarray(e, dtype=np.float32)
    if a.ndim == 2:
        a = a.reshape(-1) if a.shape[0] == 1 else a.mean(0)
    return a.reshape(-1)

def _post(payload):
    r = requests.post(f"{SERVER}/embedding", json=payload, timeout=300)
    r.raise_for_status()
    v = _vec(r.json())
    assert v.shape[0] == DIM, f"dim {v.shape[0]} != {DIM}"
    return v

def embed_text(t):
    return _post({"content": t})

def embed_image(path):
    b = base64.b64encode(open(path, "rb").read()).decode()
    return _post({"content": {"prompt_string": "<__media__>", "multimodal_data": [b]}})

TEXT_DOCS = {
    10: "A fluffy domestic cat sitting and grooming itself.",
    11: "A loyal dog playing fetch in a park.",
    12: "The Eiffel Tower, a wrought-iron lattice landmark in Paris.",
    13: "A modern hybrid passenger car parked on a street.",
    14: "The annual financial report shows company revenue grew twelve percent.",
    15: "Recipe: simmer tomatoes, garlic and basil for a marinara sauce.",
}
IMAGES = {20: "cat.jpg", 21: "dog.jpg", 22: "eiffel.jpg", 23: "car.jpg"}
LABELS = {**{i: f"TEXT  {t[:42]}" for i, t in TEXT_DOCS.items()},
          **{i: f"IMAGE {f}" for i, f in IMAGES.items()}}
QUERIES = [("a kitten", 20), ("man's best friend that barks", 21),
           ("the famous iron tower in Paris", 22), ("an automobile you can drive", 23)]

def main():
    print(f"server = {SERVER}\nembedding corpus ...")
    ids, vecs = [], []
    for i, t in TEXT_DOCS.items(): ids.append(i); vecs.append(embed_text(t))
    for i, f in IMAGES.items():    ids.append(i); vecs.append(embed_image(f"{IMG}/{f}"))
    mat = np.vstack(vecs).astype(np.float32)
    print(f"  {len(ids)} items embedded (6 text + 4 image), dim={DIM}")

    # ---- Part A: PURE cross-modal — text query against IMAGE-ONLY corpus -----
    print("\n=== A. Cross-modal: TEXT query -> IMAGE-only corpus (cosine) ===")
    img_ids = list(IMAGES); img_mat = mat[[ids.index(i) for i in img_ids]]
    img_mat = img_mat / np.linalg.norm(img_mat, axis=1, keepdims=True)
    ahits = 0
    for q, want in QUERIES:
        qv = embed_text(q); qv = qv/np.linalg.norm(qv)
        sims = sorted(zip((img_mat@qv).tolist(), img_ids), reverse=True)
        top = sims[0][1]; ahits += (top == want)
        flag = "OK" if top == want else "XX"
        print(f"  [{flag}] {q!r:34} -> " +
              " | ".join(f"{LABELS[i].split()[1]}:{s:+.3f}" for s, i in sims))
    print(f"  cross-modal accuracy: {ahits}/{len(QUERIES)}")

    # ---- Part B: unified turbovec index (text + images together) ------------
    print("\n=== B. Unified turbovec IdMapIndex(2048, 4-bit): text+image together ===")
    idx = IdMapIndex(dim=DIM, bit_width=4)
    idx.add_with_ids(mat, np.array(ids, dtype=np.uint64))
    print(f"  indexed {len(idx)} items")
    for q, want in QUERIES:
        qv = embed_text(q).reshape(1, -1)
        scores, rids = idx.search(qv, k=4)
        print(f"  {q!r}")
        for r, s in zip(rids[0], scores[0]):
            star = " *" if int(r) == want else "  "
            print(f"     {s:+.4f}{star} {LABELS[int(r)]}")

    # ---- Part C: prove delete (IdMapIndex O(1) remove, stable ids) -----------
    print("\n=== C. IdMapIndex stable-id delete ===")
    idx.remove(20)  # remove the cat IMAGE
    qv = embed_text("a kitten").reshape(1, -1)
    sc, rid = idx.search(qv, k=3)
    print(f"  after remove(20=cat image), len={len(idx)}; top ids now: {[int(x) for x in rid[0]]}")

    print(f"\nRESULT: cross-modal text->image accuracy {ahits}/{len(QUERIES)}")

if __name__ == "__main__":
    main()
