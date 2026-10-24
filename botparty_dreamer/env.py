"""Bot Party as a DreamerV3 environment (same interface as dreamerv3-torch/envs/crafter.py).

One episode = one race of P1 against the bots. The game runs in its own Godot
process (botparty_dreamer.game.launch) and talks through godot_rl_agents' GodotEnv.

Observation keys
    state        65 floats from DreamerController (see doc/implementation_log.md, step 3)
    image        what TensorBoard shows: a 64x64 minimap when `minimap`, else 1x1 (unused
                 by the model, which only reads `state`; dreamerv3-torch requires the key)
    is_first / is_last / is_terminal
    log_<mode>_{first,finished,eliminated}   1 on the last step: per-race stats in TensorBoard
"""

import gym
import numpy as np
from godot_rl.core.godot_env import GodotEnv

from botparty_dreamer.game import ROOT, launch

MOVES, JUMPS = 3, 2  # DreamerController.get_action_space(): move x jump
OUTCOMES = ("first", "finished", "eliminated")

# Layout of the 65 floats (DreamerController._features).
RAYS, DANGERS, OTHERS = 16, 6, 3
SCALE = 1000.0


class BotParty:
    metadata = {}

    def __init__(self, task="steps", mode="train", port=11008, seed=0, action_repeat=4, speedup=8.0,
                 bots=3, bot_skill="easy", window=False, minimap=False, episodes_per_launch=0, exe=None):
        """episodes_per_launch: close the game after that many races (0: never). Used for the
        windowed eval env, so its window is only open while evaluating."""
        self._mode = mode
        self._minimap = minimap
        self._episodes_per_launch = episodes_per_launch
        self._launch_args = dict(port=port, arena=task, bots=bots, bot_skill=bot_skill, headless=not window,
                                 speedup=speedup, action_repeat=action_repeat, seed=seed, exe=exe,
                                 log=str(ROOT / "runs" / f"{mode}_{port}.log"))
        self._proc = None
        self._env = None
        self._episodes = 0
        self._start()
        self._obs_size = self._env.observation_space["obs"].shape[0]
        if episodes_per_launch:
            self._stop()  # the eval window opens with the first evaluation
        self.reward_range = [-np.inf, np.inf]

    def _start(self):
        self._proc = launch(**self._launch_args)
        self._env = GodotEnv(env_path=None, port=self._launch_args["port"])
        self._episodes = 0

    def _stop(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    @property
    def observation_space(self):
        def box(shape, dtype=np.float32, low=-np.inf, high=np.inf):
            return gym.spaces.Box(low, high, shape, dtype=dtype)

        size = (64, 64, 3) if self._minimap else (1, 1, 3)
        spaces = {
            "state": box((self._obs_size,)),
            "image": box(size, np.uint8, 0, 255),
            "is_first": box((1,), np.uint8, 0, 1),
            "is_last": box((1,), np.uint8, 0, 1),
            "is_terminal": box((1,), np.uint8, 0, 1),
        }
        spaces.update({f"log_{self._mode}_{k}": box((1,)) for k in OUTCOMES})
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = gym.spaces.Discrete(MOVES * JUMPS)
        space.discrete = True
        return space

    def step(self, action):
        move, jump = int(action) % MOVES, int(action) // MOVES
        obs, reward, done, _, info = self._env.step([np.array([move]), np.array([jump])])
        done, info = bool(done[0]), info[0]
        outcome = info.get("outcome", "") if done else ""
        o = self._obs(obs[0]["obs"], last=done, terminal=done and bool(info.get("terminal", True)))
        for k in OUTCOMES:
            # "first" also counts as "finished"
            o[f"log_{self._mode}_{k}"] = np.float32(outcome == k or (k == "finished" and outcome == "first"))
        if done:
            self._episodes += 1
            if self._episodes_per_launch and self._episodes >= self._episodes_per_launch:
                self._stop()  # evaluation is over: close the window until the next one
        return o, np.float32(reward[0]), done, info

    def reset(self):
        if self._env is None:
            self._start()
        obs, _ = self._env.reset()
        return self._obs(obs[0]["obs"], first=True)

    def close(self):
        self._stop()

    def _obs(self, raw, first=False, last=False, terminal=False):
        state = np.asarray(raw, dtype=np.float32)
        return {
            "state": state,
            "image": draw_minimap(state) if self._minimap else np.zeros((1, 1, 3), np.uint8),
            "is_first": first,
            "is_last": last,
            "is_terminal": terminal,
        }


def draw_minimap(s, size=64, view=700.0):
    """What Dreamer perceives, drawn around P1 (center): walls from the rays (grey), traps (red),
    other robots (blue), the flag direction (yellow), P1 (white) and its velocity (green)."""
    img = np.full((size, size, 3), 24, np.uint8)
    k = size / (2 * view)  # game pixels -> minimap pixels
    c = size / 2

    def put(x, y, color, r=1):
        px, py = int(round(c + x * k)), int(round(c + y * k))
        img[max(0, py - r):max(0, py + r + 1), max(0, px - r):max(0, px + r + 1)] = color

    to_goal = s[0:2] * SCALE
    vel = s[2:4] * SCALE
    # walls: each ray hits at fraction * 480 px
    for i in range(RAYS):
        frac = s[7 + i]
        if frac < 1.0:
            a = 2 * np.pi * i / RAYS
            put(np.cos(a) * frac * 480, np.sin(a) * frac * 480, (150, 150, 150), 1)
    # traps: rect center offset and size
    base = 7 + RAYS
    for i in range(DANGERS):
        dx, dy, w, h = s[base + 6 * i: base + 6 * i + 4] * SCALE
        if w == 0 and h == 0:
            continue
        x0, x1 = int(c + (dx - w / 2) * k), int(c + (dx + w / 2) * k)
        y0, y1 = int(c + (dy - h / 2) * k), int(c + (dy + h / 2) * k)
        img[max(0, y0):max(0, y1 + 1), max(0, x0):max(0, x1 + 1)] = (220, 50, 50)
    # other robots
    base += 6 * DANGERS
    for i in range(OTHERS):
        dx, dy = s[base + 2 * i: base + 2 * i + 2] * SCALE
        if abs(dx) < view and abs(dy) < view:
            put(dx, dy, (80, 140, 255), 1)
    # flag: on the border when it's further than the view
    g = to_goal
    if np.abs(g).max() > view * 0.95:
        g = g / np.abs(g).max() * view * 0.95
    put(g[0], g[1], (255, 210, 60), 2)
    # P1 and where it's going
    for t in np.linspace(0, 0.3, 8):
        put(vel[0] * t, vel[1] * t, (90, 220, 110), 0)
    put(0, 0, (255, 255, 255), 1)
    return img
