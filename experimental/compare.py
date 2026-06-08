#!/usr/bin/env python3
"""
Compare three ways to embed a video for text retrieval, using the transformers
(native) path:

  native_video  - multi-frame + audio fused into ONE vector (only possible here)
  frame_only    - a single extracted frame (what the GGUF/lcovec path indexes)
  audio_only    - the extracted audio track (the other thing lcovec indexes)

For each text query it prints cosine similarity to each representation, so you
can see what each captures. Self-contained: extracts the frame + audio with
ffmpeg into a temp dir.

Usage:
  python compare.py --video clip.mp4 \
      --query "what is said in the clip" \
      --query "what is shown on screen" \
      [--fps 1.0]
"""
import argparse
import subprocess
import tempfile
import os
from embed_tf import embed_text, embed_image, embed_audio, embed_video

def _ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", action="append", required=True, help="repeatable")
    ap.add_argument("--fps", type=float, default=1.0, help="frames/sec sampled for native video")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        frame = os.path.join(tmp, "frame.jpg")
        audio = os.path.join(tmp, "audio.wav")
        _ff(["-ss", "1", "-i", a.video, "-frames:v", "1", frame])
        _ff(["-i", a.video, "-vn", "-ar", "16000", "-ac", "1", audio])

        print("embedding native video (frames + audio) ...")
        reps = {
            "native_video": embed_video(a.video, with_audio=True, fps=a.fps),
            "frame_only":   embed_image(frame),
            "audio_only":   embed_audio(audio),
        }

    print(f"\n{'query':46}" + "".join(f"{r:>14}" for r in reps))
    for q in a.query:
        qv = embed_text(q)
        row = "".join(f"{float(qv @ reps[r]):>+14.3f}" for r in reps)
        print(f"{q:46}{row}")
    print("\nNote: absolute cosines are not comparable across representations "
          "(per-modality anisotropy); read each column's spread across queries.")

if __name__ == "__main__":
    main()
