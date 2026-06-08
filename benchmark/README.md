# Benchmark

A reproducible multimodal retrieval benchmark over a content-distinct corpus
(text, markdown, images, PDFs, speech audio, video).

## Run it

```bash
pip install -r ../requirements.txt          # + the embedder running (../embedder.sh start)
python download_corpus.py                    # -> ~/bench/corpus  (needs network; see caveats)
export LCOVEC_STORE=~/bench/store
python ../lcovec.py reset
python ../lcovec.py ingest ~/bench/corpus
python run.py
```

`run.py` issues one labeled query per source item (a PDF counts as found if any
of its pages ranks) and reports per-modality top-1 plus mixed-corpus
top-1/top-3/MRR, using the shipped per-item-calibrated `z + cos` scoring.

## Reference result (90 items, 26 queries, RTX 4060 Ti)

8 text/md docs, 7 images, 4 arXiv PDFs (68 pages), 3 speech clips, 4 videos.

Within-modality top-1: text 6/8, image 6/7, pdf 3/4, audio 2/3, video 3/4 (**20/26 = 77%**).
Mixed corpus: **top-1 62%, top-3 77%, MRR 0.71**.

Reads:
- Most misses are semantically reasonable (ML text vs ML PDFs; Eiffel image vs Paris/France text; cat vs dog).
- Audio and video retrieve fine *within* their modality but get buried in a text-heavy mixed corpus (low text->audio/video cosine + ~25:1 modality imbalance). For balanced results across modalities, search per-modality and merge top-k.

## Caveats

- **Not deterministic.** The corpus is fetched live; Wikipedia's image host rate-limits/hotlink-blocks (images may arrive as tiny HTML error pages - substitute your own labeled images), and yt-dlp results depend on search ranking and a working yt-dlp. Numbers will drift run-to-run.
- Small corpus, descriptive numbers, no confidence intervals. This measures behavior, not a leaderboard.
