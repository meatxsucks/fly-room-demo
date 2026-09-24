"""Capa motora: tasas de DNs -> comandos.

FlyGymActuator mueve el cuerpo simulado con el BrainBodyBridge original de fly-brain.
MAVLinkActuator traduce lo mismo a vx / yaw_rate de un dron y arma el mensaje MAVLink
sin enviarlo a ningún lado.
"""

from typing import Protocol

import numpy as np
from pymavlink.dialects.v20 import common as mavlink2

from . import vendor_path  # noqa: F401
from brain_body_bridge import BrainBodyBridge


class RatesView:
    """Adapta un dict de tasas (Hz) a la interfaz de DNRateDecoder que espera BrainBodyBridge."""

    def __init__(self, max_rate=200.0):
        self.max_rate = max_rate
        self.rates = {}

    def get_rate(self, name):
        return self.rates.get(name, 0.0)

    def get_normalized(self, name):
        return min(self.rates.get(name, 0.0) / self.max_rate, 1.0)

    def get_pop_rate(self, name):
        return self.rates.get(name, 0.0)


class Actuator(Protocol):
    def command(self, rates: dict, visual_threat_bias: float = 0.0) -> dict: ...


def _drive_from_rates(bridge, view, rates, visual_threat_bias, dt):
    view.rates = rates
    bridge.visual_threat_bias = visual_threat_bias
    left, right = bridge.compute_drive(dt=dt)
    return float(left), float(right)


# Cuerpo FlyGym
class FlyGymActuator:
    """Convierte DNs en [drive_izq, drive_der] y los aplica al HybridTurningController."""

    def __init__(self, sim, brain_dt_s):
        self.sim = sim
        self.brain_dt_s = brain_dt_s
        self.view = RatesView()
        self.bridge = BrainBodyBridge(self.view, escape_threshold=0.3, groom_threshold=0.02)
        self.drive = np.zeros(2)

    def command(self, rates, visual_threat_bias=0.0):
        left, right = _drive_from_rates(self.bridge, self.view, rates, visual_threat_bias, self.brain_dt_s)
        self.drive = np.array([left, right])
        return {"mode": self.bridge.mode, "left_drive": left, "right_drive": right}

    def step_body(self):
        """Un paso de física con el último drive calculado."""
        return self.sim.step(self.drive)


# Stub MAVLink
MAV_FRAME_BODY_NED = 8
# Ignora posición, aceleración y yaw absoluto; usa velocidades y yaw_rate
TYPE_MASK_VEL_YAWRATE = 0b0000_0101_1100_0111


class MAVLinkActuator:
    """Traduce el mismo drive izquierda/derecha a vx (m/s) y yaw_rate (rad/s) de un dron.

    No abre conexión ni envía nada: arma y empaqueta SET_POSITION_TARGET_LOCAL_NED
    y lo guarda en last_packet, listo para mandarse por un mavutil.mavlink_connection.
    Convención: drive_izq > drive_der hace girar a la mosca a la derecha, igual que un
    yaw_rate positivo en NED.
    """

    def __init__(self, brain_dt_s, max_vx=1.0, max_yaw_rate=0.8, target_system=1, target_component=1):
        self.brain_dt_s = brain_dt_s
        self.max_vx = max_vx
        self.max_yaw_rate = max_yaw_rate
        self.target_system = target_system
        self.target_component = target_component
        self.view = RatesView()
        self.bridge = BrainBodyBridge(self.view, escape_threshold=0.3, groom_threshold=0.02)
        self.mav = mavlink2.MAVLink(None, srcSystem=255, srcComponent=190)
        self.last_packet = b""

    def command(self, rates, visual_threat_bias=0.0, time_boot_ms=0):
        left, right = _drive_from_rates(self.bridge, self.view, rates, visual_threat_bias, self.brain_dt_s)
        vx = float(np.clip((left + right) / 2.0, -0.5, 1.0) * self.max_vx)
        yaw_rate = float(np.clip(left - right, -1.0, 1.0) * self.max_yaw_rate)
        msg = self.mav.set_position_target_local_ned_encode(
            time_boot_ms, self.target_system, self.target_component,
            MAV_FRAME_BODY_NED, TYPE_MASK_VEL_YAWRATE,
            0, 0, 0,
            vx, 0, 0,
            0, 0, 0,
            0, yaw_rate,
        )
        self.last_packet = msg.pack(self.mav)
        return {"mode": self.bridge.mode, "vx": vx, "yaw_rate": yaw_rate}
