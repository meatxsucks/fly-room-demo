"""Video del demo: cámara en tercera persona + ojo compuesto, con overlay de tasas neuronales."""

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W, PANEL_H = 640, 480
BARS_H = 240
FRAME_W, FRAME_H = 2 * PANEL_W, PANEL_H + BARS_H

# (etiqueta, claves que se promedian)
BARS = [
    ("LC4 izq", ["LC4_left"]),
    ("LC4 der", ["LC4_right"]),
    ("Giant Fiber", ["GF_1", "GF_2"]),
    ("DNa01 izq", ["DNa01_left"]),
    ("DNa01 der", ["DNa01_right"]),
    ("DNa02 izq", ["DNa02_left"]),
    ("DNa02 der", ["DNa02_right"]),
    ("P9 avance", ["P9_left", "P9_right", "P9_oDN1_left", "P9_oDN1_right"]),
]
BAR_MAX_HZ = 200.0


class ChaseCamera:
    """Cámara que sigue al tórax desde atrás, con el rumbo suavizado para evitar temblores."""

    def __init__(self, physics, fly, distance=24.0, elevation=-22.0, smoothing=0.05):
        self.model = physics.model.ptr
        self.data = physics.data.ptr
        self.renderer = mujoco.Renderer(self.model, height=PANEL_H, width=PANEL_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{fly.name}/Thorax")
        self.cam.distance = distance
        self.cam.elevation = elevation
        self.smoothing = smoothing
        self._heading = None

    def render(self, heading_rad):
        target = np.degrees(heading_rad)
        if self._heading is None:
            self._heading = target
        # Interpolación por el camino angular más corto
        delta = (target - self._heading + 180.0) % 360.0 - 180.0
        self._heading += self.smoothing * delta
        self.cam.azimuth = self._heading
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def eyes_image(retina, ommatidia):
    """Dos ojos lado a lado, en gris, a partir de las lecturas (2, 721, 2)."""
    eyes = []
    for side in range(2):
        img = retina.hex_pxls_to_human_readable(ommatidia[side], color_8bit=True).max(axis=-1)
        eyes.append(img)
    both = np.concatenate([eyes[0], np.full((eyes[0].shape[0], 8), 0, np.uint8), eyes[1]], axis=1)
    pil = Image.fromarray(both).convert("RGB")
    scale = min(PANEL_W / pil.width, (PANEL_H - 30) / pil.height)
    pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.NEAREST)
    canvas = Image.new("RGB", (PANEL_W, PANEL_H), (15, 15, 18))
    canvas.paste(pil, ((PANEL_W - pil.width) // 2, 30))
    return canvas


class VideoComposer:
    def __init__(self, path, fps):
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16)
        self.font = ImageFont.load_default(size=16)
        self.font_small = ImageFont.load_default(size=14)

    def add(self, third_person, eyes, rates, info):
        frame = Image.new("RGB", (FRAME_W, FRAME_H), (20, 20, 24))
        frame.paste(Image.fromarray(third_person), (0, 0))
        frame.paste(eyes, (PANEL_W, 0))
        d = ImageDraw.Draw(frame)
        d.text((10, 8), "Tercera persona", fill=(255, 255, 255), font=self.font)
        d.text((PANEL_W + 10, 8), "Ojo compuesto (721 ommatidios por ojo)  izq | der",
               fill=(255, 255, 255), font=self.font)

        y0 = PANEL_H + 14
        bar_x, bar_w, row_h = 130, 330, 26
        for i, (label, keys) in enumerate(BARS):
            val = float(np.mean([rates.get(k, 0.0) for k in keys]))
            y = y0 + i * row_h
            d.text((10, y), label, fill=(220, 220, 220), font=self.font_small)
            d.rectangle([bar_x, y + 2, bar_x + bar_w, y + 18], outline=(90, 90, 90))
            fill_w = int(bar_w * min(val / BAR_MAX_HZ, 1.0))
            color = (230, 80, 60) if "Giant" in label or "LC4" in label else (80, 170, 230)
            if fill_w > 0:
                d.rectangle([bar_x, y + 2, bar_x + fill_w, y + 18], fill=color)
            d.text((bar_x + bar_w + 10, y), f"{val:6.1f} Hz", fill=(220, 220, 220), font=self.font_small)

        tx = 600
        lines = [
            f"t = {info['t']:5.2f} s simulados",
            f"modo: {info['mode']}",
            f"drive izq/der: {info['left_drive']:+.2f} / {info['right_drive']:+.2f}",
            f"distancia al objetivo: {info['goal_dist']:.1f} mm",
            f"MAVLink (no enviado): vx={info['vx']:+.2f} m/s  yaw_rate={info['yaw_rate']:+.2f} rad/s",
            f"escala barras: 0-{BAR_MAX_HZ:.0f} Hz",
        ]
        for i, line in enumerate(lines):
            d.text((tx, y0 + i * 30), line, fill=(235, 235, 235), font=self.font)
        self.writer.append_data(np.asarray(frame))

    def close(self):
        self.writer.close()
