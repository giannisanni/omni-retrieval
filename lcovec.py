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

# video: sample this many frames evenly across the clip, plus up to this many
# seconds of audio, and fuse them into ONE vector (must fit the server's -c 8192).
VIDEO_FRAMES   = int(os.environ.get("LCOVEC_VIDEO_FRAMES", "6"))
VIDEO_AUDIO_SEC = int(os.environ.get("LCOVEC_VIDEO_AUDIO_SEC", "45"))

def _b64(p): return base64.b64encode(open(p, "rb").read()).decode()

def embed_text(t): return _post({"content": t}, "text")
def embed_media(p): return _post({"content": {"prompt_string": "<__media__>",
                                  "multimodal_data": [_b64(p)]}}, p)

def embed_fused(paths, what):
    """Embed several media files (e.g. video frames + audio track) into one vector."""
    items = [_b64(p) for p in paths]
    return _post({"content": {"prompt_string": "<__media__>" * len(items),
                              "multimodal_data": items}}, what)

def _ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)

def _duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0

def video_parts(path):
    """Extract VIDEO_FRAMES frames spread across the clip + a clipped audio track."""
    os.makedirs(DERIV, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    dur = _duration(path) or 1.0
    frames = []
    for i in range(VIDEO_FRAMES):
        t = dur * (i + 0.5) / VIDEO_FRAMES
        fp = os.path.join(DERIV, f"{stem}.f{i}.jpg")
        try:
            _ff(["-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", fp])
            if os.path.exists(fp):
                frames.append(fp)
        except Exception:
            pass
    audio = os.path.join(DERIV, f"{stem}.audio.wav")
    try:
        _ff(["-t", str(VIDEO_AUDIO_SEC), "-i", path, "-vn", "-ar", "16000", "-ac", "1", audio])
        if not (os.path.exists(audio) and os.path.getsize(audio) > 1000):
            audio = None
    except Exception:
        audio = None
    return frames, audio

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
def _embed_file(f, m):
    """Return (vector, preview) for one file, or raise. Video is fused multi-frame + audio."""
    if m == "text":
        txt = open(f, errors="replace").read()
        return embed_text(txt[:8000]), txt[:80].replace("\n", " ")
    if m in ("image", "audio"):
        return embed_media(f), ""
    # video: several frames + audio fused into one vector
    frames, audio = video_parts(f)
    if not frames and not audio:
        raise RuntimeError("ffmpeg extracted no frames or audio")
    parts = frames + ([audio] if audio else [])
    return embed_fused(parts, f), f"[video: {len(frames)} frames{' + audio' if audio else ''}]"

def cmd_ingest(args):
    idx, meta = load_store()
    known = {v["path"] for v in meta["items"].values()}
    files = [f for f in expand(args.paths) if modality_of(f) and f not in known]
    if not files: print("nothing new to ingest."); return
    P = probe_matrix()   # for per-item baseline calibration
    nid, nvec = [], []
    for f in files:
        m = modality_of(f)
        try:
            v, preview = _embed_file(f, m)
        except Exception as e:
            print(f"  skip {os.path.basename(f)} ({m}): {e}"); continue
        i = meta["next_id"]; meta["next_id"] += 1
        bm, bs = item_baseline(v, P)
        meta["items"][str(i)] = {"path": f, "modality": m, "preview": preview, "bm": bm, "bs": bs}
        nid.append(i); nvec.append(v)
        print(f"  +{m:6} id={i}  {os.path.basename(f)}")
    if nid:
        idx.add_with_ids(np.vstack(nvec).astype(np.float32), np.array(nid, dtype=np.uint64))
        save_store(idx, meta)
    print(f"index now holds {len(idx)} items.")

def _by_mod(meta):
    by = {}
    for sid, v in meta["items"].items(): by.setdefault(v["modality"], []).append(int(sid))
    return by

# Diverse probe queries used to estimate each ITEM's baseline cosine to text.
# Subtracting that baseline closes the cross-modal gap AND cancels per-item
# anisotropy (e.g. a speech-bearing video that is a "text magnet" with high
# cosine to every query). Computed once per item at ingest; works for singletons.
PROBES = [
    "a photograph of an animal", "a city skyline at night", "a person speaking",
    "a piece of music playing", "a financial report", "a cooking recipe",
    "a landscape with mountains", "a sports event", "a technical diagram",
    "a historical landmark", "a car on a road", "a child playing",
    "computer code on a screen", "a news broadcast", "a scientific experiment",
    "a product advertisement",
]
PROBES_F = os.path.join(STORE, "probes.npy")

def _unit(v): return v / (np.linalg.norm(v) + 1e-9)

def probe_matrix():
    """Embed the probe bank once and cache it (shape: len(PROBES) x DIM, unit rows)."""
    if os.path.exists(PROBES_F):
        return np.load(PROBES_F)
    P = np.vstack([_unit(embed_text(p)) for p in PROBES]).astype(np.float32)
    os.makedirs(STORE, exist_ok=True); np.save(PROBES_F, P)
    return P

def item_baseline(vec, P):
    cs = P @ _unit(vec).astype(np.float32)
    return float(cs.mean()), float(cs.std() + 1e-9)

def cmd_query(args):
    idx, meta = load_store()
    if len(idx) == 0: sys.exit("index empty - ingest something first.")
    by = _by_mod(meta)
    qv = embed_text(args.text).reshape(1, -1).astype(np.float32)
    merged = []
    for m, ids in by.items():
        sc, rid = idx.search(qv, k=min(len(ids), max(args.k * 4, 10)),
                             allowlist=np.array(ids, dtype=np.uint64))
        for s, r in zip(sc[0], rid[0]):
            it = meta["items"][str(r)]
            bm, bs = it.get("bm", 0.0), it.get("bs", 1.0)
            z = (float(s) - bm) / bs   # how unusual is this cos for THIS item
            # light blend with raw cos: per-item z surfaces sparse/low-baseline
            # modalities (a lone video/audio), the cos term keeps strong absolute
            # matches (text) on top and damps irrelevant low-baseline creep
            merged.append((z + 1.0 * float(s), z, float(s), int(r), m))
    merged.sort(reverse=True)
    print(f"query: {args.text!r}\n")
    for blend, z, s, r, m in merged[:args.k]:
        it = meta["items"][str(r)]
        tail = it["preview"] or os.path.basename(it["path"])
        name = os.path.basename(it["path"]) if it["preview"] else ""
        print(f"  score={blend:+.2f} (z={z:+.2f} cos={s:+.3f})  [{m:5}] {tail}  {name}".rstrip())

def cmd_stats(args):
    idx, meta = load_store(); by = _by_mod(meta)
    print(f"index: {len(idx)} items, dim={DIM}, 4-bit  ({STORE})")
    for m in ("text", "image", "audio", "video"):
        if by.get(m): print(f"  {m:5}: {len(by[m])}")

def cmd_reset(args):
    for f in (IDX_F, META_F, PROBES_F):
        if os.path.exists(f): os.remove(f)
    print("store wiped.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("query");  p.add_argument("text"); p.add_argument("-k", type=int, default=6); p.set_defaults(fn=cmd_query)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    sub.add_parser("reset").set_defaults(fn=cmd_reset)
    a = ap.parse_args(); a.fn(a)
