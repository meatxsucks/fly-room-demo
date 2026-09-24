"""Capa cerebral: el modelo LIF de fly-brain sobre el conectoma FlyWire v783, sin cambios.

Expone una sola operación, step(sensory) -> dict de tasas, para que sensores y actuadores
no dependan de los detalles internos de BrainEngine.
"""

from pathlib import Path

from . import vendor_path  # noqa: F401
from brain_body_bridge import BrainEngine, DNRateDecoder, DN_NEURONS, DN_GROUPS

# Poblaciones que se reportan junto a las DNs (vienen de VisualSystem)
POPULATIONS = ("LC4_left", "LC4_right", "LPLC2_left", "LPLC2_right")


class Brain:
    """Cerebro de 138.639 neuronas LIF. Cada step() avanza un paso de 0,1 ms."""

    def __init__(self, plastic_path, stimulus="p9", window_ms=50.0):
        # plastic_path apunta a un archivo propio: si no existe se parte del conectoma v783 puro
        self.engine = BrainEngine(device="cuda", plastic_path=Path(plastic_path))
        self.decoder = DNRateDecoder(window_ms=window_ms, dt_ms=self.engine.dt, max_rate=200.0)
        self.device = self.engine.device
        self.flyid2i = self.engine.flyid2i
        self.i2flyid = self.engine.i2flyid
        self._stimulus = None
        self._visual = (None, None)
        self.set_stimulus(stimulus)

    def register_populations(self, populations):
        for name, idx in populations.items():
            self.engine.register_population(name, idx)
            self.decoder.register_population(name)

    def set_stimulus(self, name):
        # set_stimulus de fly-brain pone todas las tasas en cero, por eso se reinyecta la visión
        self._stimulus = name
        self.engine.set_stimulus(name)
        if self._visual[0] is not None:
            self.engine.set_visual_rates(*self._visual)

    def step(self, sensory=None):
        """sensory: {'visual': (índices, tasas), 'stimulus': nombre}. Devuelve tasas en Hz."""
        sensory = sensory or {}
        if "stimulus" in sensory and sensory["stimulus"] != self._stimulus:
            self.set_stimulus(sensory["stimulus"])
        if sensory.get("visual") is not None and sensory["visual"][0] is not None:
            self._visual = sensory["visual"]
            self.engine.set_visual_rates(*self._visual)

        self.engine.step()
        pops = self.engine.get_population_spikes() if self.engine.populations else None
        self.decoder.update(self.engine.get_dn_spikes(), pops)
        return self.rates()

    def rates(self):
        out = dict(self.decoder.rates)
        out.update(self.decoder.pop_rates)
        return out


__all__ = ["Brain", "DN_NEURONS", "DN_GROUPS", "POPULATIONS"]
