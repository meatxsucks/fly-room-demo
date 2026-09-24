"""Control: mismo cerebro con P9 tónico y sin entrada visual, para ver si el Giant Fiber se activa solo."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyroom.brain import Brain

STEPS = 1000

brain = Brain(plastic_path=Path("/nonexistent/sin_plasticidad_previa.pt"), stimulus="p9")
gf = np.array([np.mean([r["GF_1"], r["GF_2"]]) for r in (brain.step({"stimulus": "p9"}) for _ in range(STEPS))])

for a in range(0, STEPS, 100):
    print(f"pasos {a}-{a + 99}: GF medio {gf[a:a + 100].mean():.1f} Hz, max {gf[a:a + 100].max():.1f} Hz")
