#!/usr/bin/env bash
# Download four CC-licensed sample images (Wikimedia Commons) into ./images/
# so poc.py has something to embed. Uses the stable Special:FilePath redirect.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/images"
mkdir -p "$DIR"
get() { curl -sSL -A "Mozilla/5.0" -o "$DIR/$1" "$2" && echo "got $1"; }

get cat.jpg    "https://commons.wikimedia.org/wiki/Special:FilePath/Cat03.jpg?width=384"
get dog.jpg    "https://commons.wikimedia.org/wiki/Special:FilePath/YellowLabradorLooking_new.jpg?width=384"
get eiffel.jpg "https://commons.wikimedia.org/wiki/Special:FilePath/Tour_Eiffel_Wikimedia_Commons.jpg?width=384"
get car.jpg    "https://commons.wikimedia.org/wiki/Special:FilePath/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg?width=384"

echo "saved to $DIR"
