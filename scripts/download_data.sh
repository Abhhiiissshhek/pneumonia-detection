#!/usr/bin/env bash
# Download the Kaggle "Chest X-Ray Images (Pneumonia)" dataset into data/chest_xray.
#
# Requires the Kaggle CLI and an API token:
#   pip install kaggle
#   Put kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > Create New Token)
set -euo pipefail

DEST="${1:-data}"
mkdir -p "$DEST"

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p "$DEST"
unzip -q -o "$DEST/chest-xray-pneumonia.zip" -d "$DEST"
rm "$DEST/chest-xray-pneumonia.zip"

# The archive contains a duplicate nested copy and macOS metadata; drop them.
rm -rf "$DEST/chest_xray/chest_xray" "$DEST/chest_xray/__MACOSX"

echo "Dataset ready in $DEST/chest_xray:"
for split in train val test; do
  for cls in NORMAL PNEUMONIA; do
    printf '  %-5s %-9s %5d images\n' "$split" "$cls" "$(find "$DEST/chest_xray/$split/$cls" -type f | wc -l)"
  done
done
