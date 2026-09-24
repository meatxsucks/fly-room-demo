"""Habitación MuJoCo para la mosca: 4 paredes texturizadas, obstáculos, una luz y un objetivo."""

import numpy as np
from flygym.arena import BaseArena


# Paredes: (nombre, centro xy, tamaño medio xy, textura)
# Las texturas alternan patrones para que cada pared se distinga en el ojo compuesto.
WALL_TEXTURES = {
    "norte": dict(builtin="checker", rgb1=(0.05, 0.05, 0.05), rgb2=(0.95, 0.95, 0.95), texrepeat=(16, 1)),
    "sur": dict(builtin="checker", rgb1=(0.1, 0.1, 0.1), rgb2=(0.9, 0.9, 0.9), texrepeat=(8, 3)),
    "este": dict(builtin="gradient", rgb1=(0.95, 0.95, 0.95), rgb2=(0.1, 0.1, 0.15), texrepeat=(1, 1)),
    "oeste": dict(builtin="flat", rgb1=(0.2, 0.25, 0.45), rgb2=(0.2, 0.25, 0.45), texrepeat=(1, 1)),
}

# Obstáculos: (nombre, tipo, xy, tamaño MuJoCo, rgba)
OBSTACLES = [
    ("caja_1", "box", (14.0, 7.0), (3.0, 3.0, 4.0), (0.15, 0.12, 0.1, 1.0)),
    ("caja_2", "box", (-12.0, 16.0), (4.0, 2.5, 3.0), (0.25, 0.2, 0.15, 1.0)),
    ("pilar_1", "cylinder", (20.0, -12.0), (2.5, 6.0), (0.1, 0.1, 0.12, 1.0)),
    ("pilar_2", "cylinder", (-18.0, -14.0), (2.0, 6.0), (0.3, 0.3, 0.35, 1.0)),
]


class RoomArena(BaseArena):
    """Habitación cuadrada centrada en el origen; la mosca aparece en (0, 0) mirando a +x."""

    def __init__(self, half_size=40.0, wall_height=12.0, goal_pos=(32.0, 4.0), goal_radius=2.5):
        super().__init__()
        self.half_size = half_size
        self.wall_height = wall_height
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.goal_radius = goal_radius
        self.friction = (1, 0.005, 0.0001)
        root = self.root_element

        root.asset.add("texture", type="skybox", builtin="gradient",
                       rgb1=(0.8, 0.8, 0.82), rgb2=(0.6, 0.6, 0.65), width=256, height=256)

        # Piso de bajo contraste para que el umbral de T2 no lo confunda con objetos
        floor_tex = root.asset.add("texture", name="piso_tex", type="2d", builtin="checker",
                                   width=256, height=256, rgb1=(0.48, 0.46, 0.43), rgb2=(0.54, 0.52, 0.48))
        floor_mat = root.asset.add("material", name="piso_mat", texture=floor_tex,
                                   texrepeat=(20, 20), reflectance=0.02)
        root.worldbody.add("geom", type="plane", name="piso", material=floor_mat,
                           size=[half_size * 1.5, half_size * 1.5, 1], friction=self.friction,
                           conaffinity=0)

        h = half_size
        walls = {
            "norte": ((0.0, h), (h, 0.5)),
            "sur": ((0.0, -h), (h, 0.5)),
            "este": ((h, 0.0), (0.5, h)),
            "oeste": ((-h, 0.0), (0.5, h)),
        }
        for name, (center, half_xy) in walls.items():
            spec = WALL_TEXTURES[name]
            tex = root.asset.add("texture", name=f"pared_{name}_tex", type="2d", builtin=spec["builtin"],
                                 width=256, height=256, rgb1=spec["rgb1"], rgb2=spec["rgb2"])
            mat = root.asset.add("material", name=f"pared_{name}_mat", texture=tex,
                                 texrepeat=spec["texrepeat"], texuniform=False)
            root.worldbody.add("geom", type="box", name=f"pared_{name}", material=mat,
                               pos=(center[0], center[1], wall_height / 2),
                               size=(half_xy[0], half_xy[1], wall_height / 2),
                               friction=self.friction)

        for name, gtype, xy, size, rgba in OBSTACLES:
            z = size[-1]
            root.worldbody.add("geom", type=gtype, name=name, pos=(xy[0], xy[1], z),
                               size=size, rgba=rgba, friction=self.friction)

        # Fuente de luz principal: foco cenital con sombras
        root.worldbody.add("light", name="foco", pos=(0, 0, 60), dir=(0.1, 0.1, -1),
                           diffuse=(0.55, 0.53, 0.48), specular=(0.3, 0.3, 0.3),
                           castshadow=True, cutoff=70, exponent=2)

        # Objetivo: esfera roja emisiva, sin colisión para que la mosca pueda "alcanzarla"
        goal_mat = root.asset.add("material", name="objetivo_mat", rgba=(0.9, 0.1, 0.1, 1.0), emission=0.4)
        root.worldbody.add("geom", type="sphere", name="objetivo", size=(goal_radius,),
                           pos=(self.goal_pos[0], self.goal_pos[1], goal_radius),
                           material=goal_mat, contype=0, conaffinity=0)

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def _get_max_floor_height(self):
        return 0.0

    def distance_to_goal(self, fly_xy):
        return float(np.linalg.norm(np.asarray(fly_xy[:2]) - self.goal_pos))
