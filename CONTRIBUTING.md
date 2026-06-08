# Contributing to omni-retrieval

Thanks for your interest. This is a small, focused proof-of-concept, so contributions that keep it simple and well-documented are the most welcome: bug fixes, clearer docs, broader file-type support, retrieval-quality improvements, and packaging.

## Development setup

```bash
git clone https://github.com/giannisanni/omni-retrieval
cd omni-retrieval
pip install -r requirements.txt
```

You only need the model + the llama.cpp fork to run the tool end to end (see the README's Install section), not to lint. Linting works on a plain checkout.

## Before you open a PR

Run the same checks CI runs. Both must be clean.

```bash
# Python: correctness lint (pyflakes + syntax). See ruff.toml.
pip install ruff
ruff check .

# Shell scripts
shellcheck embedder.sh scripts/*.sh
```

If you have a GPU and the embedder set up, also sanity-check runtime behaviour:

```bash
./scripts/fetch_sample_images.sh
./embedder.sh start
./poc.py          # expect cross-modal text->image accuracy 4/4
./embedder.sh stop
```

## Code style

The code is deliberately terse (compact one-liners, multiple statements per line). **Lint is correctness-only**, not style: `ruff.toml` selects only `F` (pyflakes: undefined names, unused imports, shadowing) and `E9` (syntax errors). Please match the surrounding style rather than reformatting whole files; large stylistic diffs are hard to review and out of scope. If you want an opinionated formatter applied repo-wide, open an issue to discuss first.

## Reporting bugs

Before filing an "all my image/audio embeddings are identical" issue, confirm you launched the server with **both** required settings (this trips everyone up once):

- `LLAMA_MEDIA_MARKER='<__media__>'` (the fork randomizes the media marker)
- `--no-mmproj-offload` (the CUDA CLIP graph produces degenerate vectors)

`embedder.sh` sets both. See the README's "Two gotchas" section.

A good bug report includes: GPU and free VRAM, the `ht-llama.cpp` commit you built, your launch command, and the relevant lines from the server log (`~/.lcovec/server.log`).

## Licensing

By contributing, you agree your contributions are licensed under the project's [Apache License 2.0](LICENSE). The embedding model and third-party dependencies carry their own licenses.
