r"""Random policy against the real game: checks the Godot side (step 3) end to end.

    .venv\Scripts\python tools/smoke_env.py --episodes 3 [--window --speedup 1]
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from godot_rl.core.godot_env import GodotEnv  # noqa: E402

from botparty_dreamer.game import launch  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--port", type=int, default=11008)
p.add_argument("--window", action="store_true")
p.add_argument("--speedup", type=float, default=8, help="game speed (1 = real time)")
args = p.parse_args()

proc = launch(port=args.port, headless=not args.window, speedup=args.speedup, log=f"runs/smoke_{args.port}.log")
env = GodotEnv(env_path=None, port=args.port)
print("obs space", env.observation_space, "action space", env.action_spaces[0])
rng = np.random.default_rng(0)
try:
    for ep in range(args.episodes):
        t0 = time.time()
        obs, _ = env.reset()
        o = np.asarray(obs[0]["obs"])
        print(f"episode {ep}: first obs size={o.size} min={o.min():.2f} max={o.max():.2f} (waited {time.time() - t0:.1f}s)")
        ret, steps, done = 0.0, 0, False
        while not done:
            move, jump = rng.integers(3), rng.integers(2)
            obs, rew, done, _, info = env.step([np.array([move]), np.array([jump])])
            done = bool(done[0])
            ret += rew[0]
            steps += 1
        print(f"  steps={steps} return={ret:.2f} info={info[0]} ({time.time() - t0:.1f}s)")
finally:
    env.close()
    proc.wait(timeout=10)
