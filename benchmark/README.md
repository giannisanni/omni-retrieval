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

## Reference result (90 items, 26 queries, RTX 4060 Ti, `BLEND=2`)

8 text/md docs, 7 images, 4 arXiv PDFs (68 pages), 3 speech clips, 4 videos.

- Within-modality top-1: text 7/8, image 7/7, pdf 3/4, audio 2/3, video 3/4 (**22/26 = 85%**).
- Mixed corpus, **strict** (exact labeled item): top-1 18/26 (69%), top-3 20/26 (77%), MRR 0.74.
- Mixed corpus, **topic-graded** (right topic, any modality): top-1 **21/26 (81%)**, top-3 **23/26 (88%)**, MRR 0.86.

`run.py` reports both metrics. The corpus has deliberate cross-modal topic duplicates (coffee audio vs espresso text, etc.); strict demands the exact labeled item, topic-graded counts any right-topic item (what a user actually wants).

Reads:
- The strict/topic gap is mostly the system returning a relevant *sibling* in another modality (espresso text for the coffee query; cat photo for the cat query) - correct behaviour the strict metric penalizes.
- Audio and video retrieve fine *within* their modality but get buried in a text-heavy mixed corpus (low text->audio/video cosine + ~25:1 imbalance). Raising `BLEND` sharpens text/image but does not rescue them; for balanced cross-modal results, search per-modality and merge top-k, or lower `LCOVEC_BLEND`.

## Caveats

- **Not deterministic.** The corpus is fetched live; Wikipedia's image host rate-limits/hotlink-blocks (images may arrive as tiny HTML error pages - substitute your own labeled images), and yt-dlp results depend on search ranking and a working yt-dlp. Numbers will drift run-to-run.
- Small corpus, descriptive numbers, no confidence intervals. This measures behavior, not a leaderboard.
