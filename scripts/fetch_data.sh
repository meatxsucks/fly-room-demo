#!/usr/bin/env bash
# Descarga los datos del conectoma FlyWire v783 que usa fly-brain, fijados al commit vendorizado.
set -euo pipefail

COMMIT="27cec28d5d202eb004683fb4c1a1033eec8deea0"
BASE="https://raw.githubusercontent.com/erojasoficial-byte/fly-brain/${COMMIT}/data"
DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor/fly_brain/data"

mkdir -p "$DEST"
for f in 2025_Completeness_783.csv 2025_Connectivity_783.parquet flywire_annotations.tsv; do
  if [ -s "$DEST/$f" ]; then
    echo "ya existe: $f"
  else
    echo "descargando: $f"
    curl -fL --retry 3 -o "$DEST/$f" "$BASE/$f"
  fi
done
