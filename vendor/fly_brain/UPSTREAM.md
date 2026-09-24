# Código de terceros

Copiado sin modificaciones de https://github.com/erojasoficial-byte/fly-brain
en el commit `27cec28d5d202eb004683fb4c1a1033eec8deea0` (2026-03-21), bajo licencia MIT (ver `LICENSE`).

- `brain_body_bridge.py`: BrainEngine (LIF + plasticidad Hebbiana), DNRateDecoder, BrainBodyBridge
- `visual_system.py`: mapeo ommatidios -> neuronas visuales del conectoma
- `code/run_pytorch.py`, `code/benchmark.py`: modelo LIF en PyTorch y rutas de datos

Los datos (`data/`) no se versionan; se descargan con `scripts/fetch_data.sh`.
