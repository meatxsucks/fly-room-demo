"""Capa sensorial: frames RGB de cada ojo -> ommatidios -> tasas de entrada al conectoma.

La fuente de frames es intercambiable. Hoy los frames salen de las cámaras de ojo de MuJoCo;
para una cámara real basta con otra clase que entregue el mismo par de imágenes RGB.
"""

from typing import Protocol

import mujoco
import numpy as np

from . import vendor_path  # noqa: F401
from visual_system import VisualSystem

# Resolución cruda que espera la Retina de FlyGym (filas, columnas)
EYE_FRAME_SHAPE = (512, 450)


class FrameSource(Protocol):
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (rgb_izq, rgb_der), cada uno uint8 de forma (512, 450, 3)."""


# Fuente MuJoCo
class MujocoEyeCameras:
    """Renderiza las cámaras de los ojos del modelo FlyGym, ocultando las geoms propias."""

    def __init__(self, physics, fly):
        self.model = physics.model.ptr
        self.data = physics.data.ptr
        self.renderer = mujoco.Renderer(self.model, height=EYE_FRAME_SHAPE[0], width=EYE_FRAME_SHAPE[1])
        self.cam_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"{fly.name}/{side}Eye_cam")
            for side in ("L", "R")
        ]
        self.hide_ids = [
            gid for gid in (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{fly.name}/{g}")
                for g in getattr(fly, "_geoms_to_hide", [])
            ) if gid >= 0
        ]

    def read(self):
        alpha = self.model.geom_rgba[self.hide_ids, 3].copy()
        self.model.geom_rgba[self.hide_ids, 3] = 0.0
        frames = []
        for cid in self.cam_ids:
            self.renderer.update_scene(self.data, camera=cid)
            frames.append(self.renderer.render().copy())
        self.model.geom_rgba[self.hide_ids, 3] = alpha
        return frames[0], frames[1]


# Codificador frame -> ommatidios -> tasas
class CompoundEyeEncoder:
    """Convierte frames RGB en lecturas de ommatidios y en tasas de entrada para el cerebro.

    Las tasas vienen de VisualSystem de fly-brain sin cambios: solo inyecta la vía OFF (T2),
    que por el conectoma llega a LC4 y al Giant Fiber.
    """

    def __init__(self, flyid2i, i2flyid, retina):
        self.retina = retina
        self.visual = VisualSystem(flyid2i, i2flyid)

    def frames_to_ommatidia(self, rgb_left, rgb_right):
        readouts = [
            self.retina.raw_image_to_hex_pxls(np.ascontiguousarray(self.retina.correct_fisheye(img)))
            for img in (rgb_left, rgb_right)
        ]
        return np.array(readouts, dtype=np.float32)

    def encode(self, rgb_left, rgb_right):
        """Devuelve (ommatidios (2, 721, 2), índices de neuronas, tasas Hz)."""
        omm = self.frames_to_ommatidia(rgb_left, rgb_right)
        idx, rates = self.visual.process_visual_layers(omm)
        return omm, idx, rates

    def threat_bias(self, rates):
        """Asimetría T2 derecha-izquierda, el respaldo direccional que usa fly_embodied.py."""
        eye = self.visual._T2_eye
        left = float(rates[eye == 0].mean()) if (eye == 0).any() else 0.0
        right = float(rates[eye == 1].mean()) if (eye == 1).any() else 0.0
        return (right - left) / (left + right + 1e-6)

    def population_indices(self, flyid2i):
        return {**self.visual.get_lplc2_indices(flyid2i), **self.visual.get_lc4_indices(flyid2i)}
