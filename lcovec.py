#!/usr/bin/env python3
"""
lcovec - local cross-modal retrieval over text / image / audio / video.

  embedder : LCO-Embedding-Omni-3B (ht-llama.cpp fork) on :8090, ONE 2048-d space
  index    : turbovec IdMapIndex (4-bit TurboQuant, CPU), stable uint64 ids
  bias fix : per-modality z-score + raw-cos blend via turbovec allowlist
  video    : .mp4/.mov are decomposed by ffmpeg into a frame (image) + audio track

Usage:
  lcovec.py ingest <dir-or-file> [more...]
  lcovec.py query "<text>" [-k N]
  lcovec.py stats | reset
Requires the embedder running:  ./embedder.sh start

Environment:
  LCO_SERVER     embedder URL (default http://127.0.0.1:8090)
  LCOVEC_STORE   index/metadata directory (default ~/.lcovec/store)
"""
import os, sys, json, base64, argparse, glob, subprocess
import numpy as np, requests
from turbovec import IdMapIndex

SERVER = os.environ.get("LCO_SERVER", "http://127.0.0.1:8090")
DIM    = 2048
STORE  = os.path.expanduser(os.environ.get("LCOVEC_STORE", "~/.lcovec/store"))
IDX_F  = os.path.join(STORE, "index.tvim")
META_F = os.path.join(STORE, "meta.json")
DERIV  = os.path.join(STORE, "derived")

TEXT_EXT  = {".txt", ".md", ".markdown", ".text", ".rst"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
def modality_of(path):
    e = os.path.splitext(path)[1].lower()
    return ("text" if e in TEXT_EXT else "image" if e in IMAGE_EXT else
            "audio" if e in AUDIO_EXT else "video" if e in VIDEO_EXT else None)

# ---- embedding --------------------------------------------------------------
def _vec(resp):
    e = resp[0]["embedding"] if isinstance(resp, list) else resp["embedding"]
    a = np.asarray(e, dtype=np.float32)
    return a.reshape(-1) if a.ndim == 1 else a.mean(0)

def _post(payload, what):
    try:
        r = requests.post(f"{SERVER}/embedding", json=payload, timeout=900)
    except requests.exceptions.ConnectionError:
        sys.exit(f"ERROR: embedder unreachable at {SERVER}. Run: ./embedder.sh start")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {r.text[:140]}")
    v = _vec(r.json())
    if v.shape[0] != DIM: raise RuntimeError(f"dim {v.shape[0]} != {DIM}")
    return v.astype(np.float32)

def embed_text(t): return _post({"content": t}, "text")
def embed_media(p): return _post({"content": {"prompt_string": "<__media__>",
                    "multimodal_data": [base64.b64encode(open(p, "rb").read()).decode()]}}, p)

def _ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)

def derive(path, modality):
    """Return list of sub-items to embed: {modality, kind('text'|'media'), payload, preview, from_video}."""
    if modality == "text":
        txt = open(path, errors="replace").read()
        return [{"modality": "text", "kind": "text", "payload": txt[:8000],
                 "preview": txt[:80].replace("\n", " ")}]
    if modality in ("image", "audio"):
        return [{"modality": modality, "kind": "media", "payload": path, "preview": ""}]
    # video -> frame (image) + audio (audio)
    os.makedirs(DERIV, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    frame = os.path.join(DERIV, f"{stem}.frame.jpg")
    audio = os.path.join(DERIV, f"{stem}.audio.wav")
    items = []
    try:
        _ff(["-ss", "1", "-i", path, "-frames:v", "1", frame])
        items.append({"modality": "image", "kind": "media", "payload": frame,
                      "preview": "[video frame]", "from_video": path})
    except Exception as e:
        print(f"    (no frame from {os.path.basename(path)}: {e})")
    try:
        _ff(["-i", path, "-vn", "-ar", "16000", "-ac", "1", audio])
        if os.path.getsize(audio) > 1000:
            items.append({"modality": "audio", "kind": "media", "payload": audio,
                          "preview": "[video audio]", "from_video": path})
    except Exception as e:
        print(f"    (no audio from {os.path.basename(path)}: {e})")
    return items

# ---- store ------------------------------------------------------------------
def load_store():
    os.makedirs(STORE, exist_ok=True)
    meta = json.load(open(META_F)) if os.path.exists(META_F) else {"next_id": 1, "items": {}}
    idx = IdMapIndex.load(IDX_F) if os.path.exists(IDX_F) else IdMapIndex(dim=DIM, bit_width=4)
    return idx, meta
def save_store(idx, meta): idx.write(IDX_F); json.dump(meta, open(META_F, "w"), indent=0)

def expand(paths):
    out = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for root, _, fs in os.walk(p): out += [os.path.join(root, f) for f in fs]
        else: out += glob.glob(p)
    return sorted(set(out))

# ---- commands ---------------------------------------------------------------
def cmd_ingest(args):
    idx, meta = load_store()
    known = {v.get("source") or v["path"] for v in meta["items"].values()}
    files = [f for f in expand(args.paths) if modality_of(f) and f not in known]
    if not files: print("nothing new to ingest."); return
    nid, nvec = [], []
    for f in files:
        for it in derive(f, modality_of(f)):
            try:
                v = embed_text(it["payload"]) if it["kind"] == "text" else embed_media(it["payload"])
            except Exception as e:
                print(f"  skip {os.path.basename(f)} ({it['modality']}): {e}"); continue
            i = meta["next_id"]; meta["next_id"] += 1
            meta["items"][str(i)] = {"path": f, "source": f, "modality": it["modality"],
                                     "preview": it["preview"], "from_video": it.get("from_video")}
            nid.append(i); nvec.append(v)
            tag = f"{it['modality']}" + ("/vid" if it.get("from_video") else "")
            print(f"  +{tag:9} id={i}  {os.path.basename(f)}")
    if nid:
        idx.add_with_ids(np.vstack(nvec).astype(np.float32), np.array(nid, dtype=np.uint64))
        save_store(idx, meta)
    print(f"index now holds {len(idx)} items.")

def _by_mod(meta):
    by = {}
    for sid, v in meta["items"].items(): by.setdefault(v["modality"], []).append(int(sid))
    return by

def cmd_query(args):
    idx, meta = load_store()
    if len(idx) == 0: sys.exit("index empty - ingest something first.")
    qv = embed_text(args.text).reshape(1, -1).astype(np.float32)
    merged = []
    for m, ids in _by_mod(meta).items():
        sc, rid = idx.search(qv, k=min(len(ids), max(args.k * 4, 10)),
                             allowlist=np.array(ids, dtype=np.uint64))
        sc, rid = sc[0], rid[0]
        z = (sc - sc.mean()) / (sc.std() + 1e-9) if len(sc) > 1 else sc * 0
        for s, zi, r in zip(sc, z, rid):
            merged.append((float(zi) + 3.0 * float(s), float(zi), float(s), int(r), m))
    merged.sort(reverse=True)
    print(f"query: {args.text!r}\n")
    for blend, zi, s, r, m in merged[:args.k]:
        it = meta["items"][str(r)]
        tail = it["preview"] or os.path.basename(it["path"])
        src = f"  (from {os.path.basename(it['from_video'])})" if it.get("from_video") else ""
        print(f"  score={blend:+.2f} (z={zi:+.2f} cos={s:+.3f})  [{m:5}] {tail}{src}")

def cmd_stats(args):
    idx, meta = load_store(); by = _by_mod(meta)
    print(f"index: {len(idx)} items, dim={DIM}, 4-bit  ({STORE})")
    for m in ("text", "image", "audio", "video"):
        if by.get(m): print(f"  {m:5}: {len(by[m])}")

def cmd_reset(args):
    for f in (IDX_F, META_F):
        if os.path.exists(f): os.remove(f)
    print("store wiped.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("query");  p.add_argument("text"); p.add_argument("-k", type=int, default=6); p.set_defaults(fn=cmd_query)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    sub.add_parser("reset").set_defaults(fn=cmd_reset)
    a = ap.parse_args(); a.fn(a)
