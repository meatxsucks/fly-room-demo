"""Demo: mosca con cerebro conectómico (FlyWire v783) navegando una habitación 3D.

Corre N segundos simulados, guarda el video lado a lado, un CSV con la trayectoria y las
tasas neuronales, y un JSON con FPS de simulación y uso de memoria.
"""

import argparse
import csv
import json
import platform
import resource
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from dm_control.rl.control import PhysicsError
from flygym import Fly
from flygym.examples.locomotion.turning_controller import HybridTurningController

from flyroom.actuators import FlyGymActuator, MAVLinkActuator
from flyroom.arena import RoomArena
from flyroom.brain import Brain
from flyroom.render import ChaseCamera, VideoComposer, eyes_image
from flyroom.sensors import CompoundEyeEncoder, MujocoEyeCameras

# Mismas proporciones que fly_embodied.py de fly-brain
BODY_DT = 1e-4
BRAIN_RATIO = 100
VISION_RATIO = 1000

# Recuperaciones permitidas tras un PhysicsError antes de cortar la corrida
MAX_RECOVERIES = 50

LOG_KEYS = ["GF_1", "GF_2", "LC4_left", "LC4_right", "LPLC2_left", "LPLC2_right", "DNa01_left", "DNa01_right",
            "DNa02_left", "DNa02_right", "P9_left", "P9_right", "MDN_1", "aDN1_left"]


# Instantánea de integración completa (tiempo, qpos, qvel, act, warmstart, controles, mocap)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def physics_snapshot(physics):
    model, data = physics.model.ptr, physics.data.ptr
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def physics_restore(physics, state):
    model, data = physics.model.ptr, physics.data.ptr
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    # MuJoCo se reinicia solo ante BADQACC y deja el contador en 1; si no se limpia,
    # dm_control deja de ver los reinicios siguientes y la mosca se teletransporta sin aviso
    for w in data.warning:
        w.number = 0
    mujoco.mj_forward(model, data)


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reporta bytes, Linux kilobytes
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3


def gpu_report():
    if torch.cuda.is_available():
        return {"device": torch.cuda.get_device_name(0),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6}
    return {"device": None,
            "nota": "sin CUDA; BrainEngine solo usa cuda o cpu, así que el cerebro corre en CPU",
            "mps_disponible": torch.backends.mps.is_available()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=30.0, help="segundos simulados")
    ap.add_argument("--fps", type=int, default=30, help="fps del video")
    ap.add_argument("--stimulus", default="p9", help="estímulo tónico de fly-brain (p9 = caminar)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0, help="semilla de torch para la entrada Poisson del cerebro")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    brain = Brain(plastic_path=out / "plastic_weights_room.pt", stimulus=args.stimulus)
    arena = RoomArena()
    # HybridTurningController exige contacto en tibia y tarsos para detectar tropiezos
    contact_sensors = [f"{leg}{seg}" for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
                       for seg in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]]
    fly = Fly(enable_adhesion=True, draw_adhesion=False, enable_vision=True,
              contact_sensor_placements=contact_sensors)
    sim = HybridTurningController(fly=fly, timestep=BODY_DT, seed=0, arena=arena)
    # Igual que fly_embodied: la visión se renderiza a mano, no dentro de sim.step
    fly.enable_vision = False
    obs, _ = sim.reset(seed=0)

    eye_encoder = CompoundEyeEncoder(brain.flyid2i, brain.i2flyid, retina=fly.retina)
    brain.register_populations(eye_encoder.population_indices(brain.flyid2i))
    eyes = MujocoEyeCameras(sim.physics, fly)
    chase = ChaseCamera(sim.physics, fly)
    body = FlyGymActuator(sim, brain_dt_s=BRAIN_RATIO * BODY_DT)
    drone = MAVLinkActuator(brain_dt_s=BRAIN_RATIO * BODY_DT)
    video = VideoComposer(str(out / "room_demo.mp4"), fps=args.fps)

    t_setup = time.perf_counter() - t_start
    print(f"[demo] setup {t_setup:.1f} s, cerebro en {brain.device}")

    n_steps = int(round(args.duration / BODY_DT))
    frame_every = int(round(1.0 / (args.fps * BODY_DT)))
    timers = {"cerebro": 0.0, "fisica": 0.0, "vision": 0.0, "video": 0.0}
    rates = brain.rates()
    sensory = {"stimulus": args.stimulus}
    ommatidia = np.zeros((2, 721, 2), np.float32)
    threat_bias = 0.0
    body_cmd = {"mode": "walking", "left_drive": 0.0, "right_drive": 0.0}
    drone_cmd = {"vx": 0.0, "yaw_rate": 0.0}
    goal_reached_at = None
    min_goal_dist = np.inf
    physics_error = None
    physics_recoveries = []
    silent_resets = []
    last_physics_time = 0.0
    snapshot = None
    log_rows = []
    frames = 0

    t_loop = time.perf_counter()
    step = 0
    for step in range(n_steps):
        t_sim = step * BODY_DT

        if step % VISION_RATIO == 0:
            snapshot = physics_snapshot(sim.physics)
            t0 = time.perf_counter()
            ommatidia, idx, vis_rates = eye_encoder.encode(*eyes.read())
            sensory["visual"] = (idx, vis_rates)
            threat_bias = eye_encoder.threat_bias(vis_rates) if vis_rates is not None else 0.0
            timers["vision"] += time.perf_counter() - t0

        if step % BRAIN_RATIO == 0:
            t0 = time.perf_counter()
            rates = brain.step(sensory)
            sensory.pop("visual", None)
            timers["cerebro"] += time.perf_counter() - t0
            body_cmd = body.command(rates, threat_bias)
            drone_cmd = drone.command(rates, threat_bias, time_boot_ms=int(t_sim * 1000))

        t0 = time.perf_counter()
        try:
            obs, *_ = body.step_body()
        except PhysicsError as exc:
            # Workaround: volver al último estado físico válido (hasta 100 ms atrás) y seguir.
            # El estado interno del CPG y del cerebro no se rebobina.
            physics_recoveries.append(round(t_sim, 4))
            print(f"[demo] PhysicsError en t={t_sim:.3f} s ({exc}); restaurando instantánea")
            if len(physics_recoveries) > MAX_RECOVERIES or snapshot is None:
                physics_error = f"{exc} (t={t_sim:.3f} s)"
                print("[demo] demasiadas recuperaciones, se corta la simulación")
                break
            physics_restore(sim.physics, snapshot)
            last_physics_time = float(sim.physics.data.time)
            continue
        timers["fisica"] += time.perf_counter() - t0
        physics_time = float(sim.physics.data.time)
        if physics_time < last_physics_time:
            silent_resets.append(round(t_sim, 4))
            print(f"[demo] reinicio de MuJoCo sin excepción en t={t_sim:.3f} s")
        last_physics_time = physics_time

        pos = obs["fly"][0]
        orient = obs["fly_orientation"]
        heading = float(np.arctan2(orient[1], orient[0]))
        goal_dist = arena.distance_to_goal(pos)
        min_goal_dist = min(min_goal_dist, goal_dist)
        if goal_reached_at is None and goal_dist < arena.goal_radius + 1.5:
            goal_reached_at = t_sim
            print(f"[demo] objetivo alcanzado en t={t_sim:.2f} s")

        if step % frame_every == 0:
            t0 = time.perf_counter()
            info = {"t": t_sim, "goal_dist": goal_dist, **body_cmd, **drone_cmd}
            video.add(chase.render(heading), eyes_image(eye_encoder.retina, ommatidia), rates, info)
            frames += 1
            timers["video"] += time.perf_counter() - t0
            log_rows.append([round(t_sim, 4), *np.round(pos, 3), round(heading, 4), body_cmd["mode"],
                             round(body_cmd["left_drive"], 3), round(body_cmd["right_drive"], 3),
                             round(drone_cmd["vx"], 3), round(drone_cmd["yaw_rate"], 3),
                             *[round(rates.get(k, 0.0), 2) for k in LOG_KEYS]])

        if step % 10000 == 0:
            wall = time.perf_counter() - t_loop
            print(f"[demo] t={t_sim:5.1f}s  pos=({pos[0]:6.1f},{pos[1]:6.1f})  modo={body_cmd['mode']:<8} "
                  f"GF={np.mean([rates.get('GF_1', 0), rates.get('GF_2', 0)]):5.1f}Hz  "
                  f"objetivo={goal_dist:5.1f}mm  pared={wall:6.1f}s")

    video.close()
    wall_loop = time.perf_counter() - t_loop
    sim_seconds = (step + 1) * BODY_DT

    with open(out / "room_demo_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "heading", "modo", "drive_izq", "drive_der", "vx", "yaw_rate", *LOG_KEYS])
        w.writerows(log_rows)

    metrics = {
        "segundos_simulados": round(sim_seconds, 3),
        "pasos_fisica": step + 1,
        "pasos_cerebro": (step // BRAIN_RATIO) + 1,
        "frames_video": frames,
        "setup_s": round(t_setup, 1),
        "loop_s": round(wall_loop, 1),
        "pasos_fisica_por_s": round((step + 1) / wall_loop, 1),
        "factor_tiempo_real": round(sim_seconds / wall_loop, 4),
        "frames_video_por_s_pared": round(frames / wall_loop, 2),
        "tiempo_por_componente_s": {k: round(v, 1) for k, v in timers.items()},
        "ram_pico_mb": round(peak_rss_mb(), 0),
        "gpu": gpu_report(),
        "seed": args.seed,
        "torch": torch.__version__,
        "threads_torch": torch.get_num_threads(),
        "objetivo_alcanzado_en_s": goal_reached_at,
        "distancia_minima_al_objetivo_mm": round(float(min_goal_dist), 2),
        "posicion_final_mm": [round(float(v), 2) for v in pos],
        "tiempo_fisico_mujoco_s": round(float(sim.physics.data.time), 3),
        "recuperaciones_fisica_t_s": physics_recoveries,
        "reinicios_silenciosos_t_s": silent_resets,
        "error_fisica": physics_error,
    }
    (out / "room_demo_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
