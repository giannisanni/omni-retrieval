#!/usr/bin/env python3
"""
Download a diverse, content-distinct benchmark corpus into $BENCH_ROOT
(default ~/bench/corpus): text + images (Wikipedia REST), markdown, PDFs (arXiv),
speech audio + videos (yt-dlp).

Caveats: Wikipedia's image host rate-limits/hotlink-blocks; if images come back
as tiny HTML error pages, substitute your own labeled images. yt-dlp needs
network and a recent version. Results in the README vary run-to-run.
"""
import os, shutil, subprocess, requests

ROOT = os.path.expanduser(os.environ.get("BENCH_ROOT", "~/bench/corpus"))
for d in ("text", "images", "pdf", "audio", "video"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
H = {"User-Agent": "omni-retrieval-benchmark/1.0 (research)", "Referer": "https://en.wikipedia.org/"}
YT = shutil.which("yt-dlp") or os.path.expanduser("~/.local/bin/yt-dlp")

TEXT_TOPICS = ["Photosynthesis", "French_Revolution", "Black_hole", "Espresso",
               "Basketball", "Roman_Empire", "Machine_learning", "Volcano"]
IMAGE_TOPICS = ["Cat", "Dog", "Eiffel_Tower", "Car", "Airplane", "Mountain", "Sailing_ship"]

def summary(t):
    return requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{t}", headers=H, timeout=30).json()

def is_img(b): return b[:3] == b"\xff\xd8\xff" or b[:8] == b"\x89PNG\r\n\x1a\n"

for t in TEXT_TOPICS:
    try:
        txt = summary(t).get("extract", "")
        if len(txt) < 50: print(f"  text FAIL {t}"); continue
        ext = "md" if t in ("Machine_learning", "Volcano") else "txt"
        open(f"{ROOT}/text/{t}.{ext}", "w").write(f"# {t.replace('_',' ')}\n\n{txt}" if ext == "md" else txt)
        print(f"  text OK {t}.{ext}")
    except Exception as e:
        print(f"  text FAIL {t}: {e}")

for t in IMAGE_TOPICS:
    try:
        j = summary(t)
        cands = [(j.get("originalimage") or {}).get("source"), (j.get("thumbnail") or {}).get("source")]
        for url in [u for u in cands if u]:
            b = requests.get(url, headers=H, timeout=30).content
            if is_img(b):
                ext = "png" if b[:4] == b"\x89PNG" else "jpg"
                open(f"{ROOT}/images/{t}.{ext}", "wb").write(b); print(f"  image OK {t}.{ext}"); break
        else:
            print(f"  image FAIL {t} (host blocked? substitute a local image)")
    except Exception as e:
        print(f"  image FAIL {t}: {e}")

PDFS = {"attention_1706.03762": "1706.03762", "resnet_1512.03385": "1512.03385",
        "bert_1810.04805": "1810.04805", "turboquant_2504.19874": "2504.19874"}
for name, aid in PDFS.items():
    try:
        b = requests.get(f"https://arxiv.org/pdf/{aid}", headers=H, timeout=60).content
        if b[:4] == b"%PDF":
            open(f"{ROOT}/pdf/{name}.pdf", "wb").write(b); print(f"  pdf OK {name}.pdf")
        else: print(f"  pdf FAIL {name}")
    except Exception as e:
        print(f"  pdf FAIL {name}: {e}")

def ytdl(query, tmpl, fmt, secs, reencode):
    subprocess.run([YT, "-q", "-f", fmt, "--no-playlist", "--match-filter", "duration<240",
                    "--max-downloads", "1", "-o", tmpl, f"ytsearch8:{query}"], timeout=300)
    raw = next((os.path.join(os.path.dirname(tmpl), f) for f in os.listdir(os.path.dirname(tmpl))
                if f.startswith(os.path.basename(tmpl).split('.')[0] + "_raw")), None)
    return raw

AUDIO = {"coffee": "how espresso coffee is made explained",
         "blackhole": "black holes explained documentary narration",
         "basketball": "basketball rules explained narration"}
for name, q in AUDIO.items():
    try:
        raw = ytdl(q, f"{ROOT}/audio/{name}_raw.%(ext)s", "bestaudio/best", 30, True)
        if not raw: print(f"  audio FAIL {name}"); continue
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", "30", "-i", raw,
                        "-ar", "16000", "-ac", "1", f"{ROOT}/audio/{name}.wav"], timeout=120)
        os.remove(raw); print(f"  audio OK {name}.wav")
    except Exception as e:
        print(f"  audio FAIL {name}: {e}")

VIDEO = {"cat": "close up cat meowing", "train": "train passing railway crossing",
         "cooking": "frying vegetables in a pan", "guitar": "person playing acoustic guitar"}
for name, q in VIDEO.items():
    try:
        raw = ytdl(q, f"{ROOT}/video/{name}_raw.%(ext)s", "mp4/best[ext=mp4]/best", 15, True)
        if not raw: print(f"  video FAIL {name}"); continue
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", "15", "-i", raw,
                        "-c:v", "libx264", "-c:a", "aac", f"{ROOT}/video/{name}.mp4"], timeout=180)
        os.remove(raw); print(f"  video OK {name}.mp4")
    except Exception as e:
        print(f"  video FAIL {name}: {e}")
print("DONE ->", ROOT)
