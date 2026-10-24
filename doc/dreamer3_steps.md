# DreamerV3 vs Bot Party — Steps

Goal: train a DreamerV3 agent that plays **Bot Party** (Party mode) as **P1**, against
**3 built-in bots**, and wins the race to the flag more often than they do.

| Repo | Path | Role |
|---|---|---|
| Bot Party (the game) | `C:/git/bot-party` | Godot 4.7 game: we add an RL controller on a branch |
| godot_rl_agents | `C:/git/godot_rl_agents` | Python ↔ Godot bridge (`GodotEnv`) + Godot plugin (`Sync`, `AIController2D`) |
| dreamerv3-torch | `C:/git/dreamerv3-torch` | DreamerV3 (PyTorch) |
| **bot-party-dreamer** (this repo) | `C:/git/bot-party-dreamer` | Glue: Dreamer env wrapper, configs, scripts, docs |

```
 Python (dreamerv3-torch)                         Godot (bot-party, branch "dreamer")
 ┌──────────────────────────┐   TCP :11008   ┌──────────────────────────────────────┐
 │ dreamer.py               │◄──────────────►│ Sync node (godot_rl_agents plugin)   │
 │  └ envs/botparty.py      │  obs, reward,  │  └ DreamerController (AIController2D)│
 │     └ godot_rl GodotEnv  │  done / action │      drives slot P1 via Input.action_*│
 └──────────────────────────┘                │ PartyBot x3 (slots 2-4, opponents)   │
                                             └──────────────────────────────────────┘
```

---

## 1. Game setup: 1 player (Dreamer) + 3 bots

In the game, from **Players**:

1. Join **only P1** (press Space). P1 is the slot Dreamer will drive.
2. Set **Bots = 3** (left / right on the bots row) and the skill (up / down):
   start with **Easy**, move to **Normal** then **Hard** as the agent improves.
3. Bots fill the free slots after the people (`ui/main_menu.gd:901`), so they take
   **P2, P3, P4**. Each is a `PartyBot` (`party/party_bot.gd`) that presses its own
   slot's input actions, i.e. it follows exactly the same rules as Dreamer.

The same setup from the command line (`core/command_line.gd`), which is what training uses:

```powershell
godot --path C:/git/bot-party -- --party --arena=steps --players=1 --bots=3 --bot-skill=easy
```

`--players=1` → `GameState.joined = [1]`; `--bots=3` → `Party.bot_count = 3`
(saved in the party session as `session.bots`, read back by `Party.bots()` and spawned
in `party_controller.gd:78`).

> Party mode round = **PICK** (choose an item) → **PLACE** (put it on the course) →
> **COUNTDOWN** → **RUN** (race to the flag) → **SCORE**
> (`POINTS_FINISH=10`, `POINTS_FIRST=5`, `POINTS_TRAP=2`, `POINTS_COIN=5`, first to 50).

---

## 2. godot_rl_agents (cloned locally at `C:/git/godot_rl_agents`)

### 2.1 Fetch the Godot plugin (submodule is empty)

`godot_rl_agents_plugin/` is a git submodule and is **empty** in the local clone. Its URL
is SSH (`git@github.com:...`); use HTTPS if no SSH key is set up:

```powershell
cd C:/git/godot_rl_agents
git config submodule.godot_rl_agents_plugin.url https://github.com/edbeeching/godot_rl_agents_plugin.git
git submodule update --init
```

### 2.2 Install the Python package

Use one virtualenv (Python 3.11) for both Dreamer and godot_rl:

```powershell
cd C:/git/bot-party-dreamer
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e C:/git/godot_rl_agents
pip install -r C:/git/dreamerv3-torch/requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build
```

> `dreamerv3-torch` pins `gym`; `godot_rl` uses `gymnasium`. Both can coexist; the
> wrapper in step 5 only exposes `gym` spaces to Dreamer.

### 2.3 Add the plugin to the game (on a branch, not on `main`)

Keep your son's `main` untouched:

```powershell
cd C:/git/bot-party
git switch -c dreamer
xcopy /E /I C:\git\godot_rl_agents\godot_rl_agents_plugin\addons\godot_rl_agents addons\godot_rl_agents
```

Then in the Godot editor: **Project → Project Settings → Plugins → enable "GodotRLAgents"**.

---

## 3. Godot side: the Dreamer controller (`bot-party`, branch `dreamer`)

### 3.1 How Dreamer presses keys

Copy the trick `PartyBot` uses (`party_bot.gd`, `_ready`): erase the keyboard events of
slot 1 so the keyboard no longer moves P1, then press / release the actions in code:

```gdscript
Input.action_press("p1_move_right")
Input.action_release("p1_move_left")
```

`player/player.gd` reads `p1_move_left/right`, `p1_action_1` (jump), `p1_move_up` (jump),
`p1_action_2` (turn), `p1_action_3` (flip) — nothing to change in the player.

### 3.2 `rl/dreamer_controller.gd` (extends `AIController2D`)

Discrete action space, kept small so Dreamer learns fast:

| key | size | meaning |
|---|---|---|
| `move` | 3 | none / left / right |
| `jump` | 2 | released / held (held is needed for long jumps) |

(Add `action_2` / `action_3` later if the arenas need them.)

```gdscript
extends AIController2D
## Dreamer's hands on slot P1. Same inputs as a human or a PartyBot.

var slot := 1
var ctl: Node            # the PartyController
var _last_dist := INF

func get_action_space() -> Dictionary:
	return {
		"move": {"size": 3, "action_type": "discrete"},
		"jump": {"size": 2, "action_type": "discrete"},
	}

func set_action(action) -> void:
	var m := int(action["move"])
	_press("move_left", m == 1)
	_press("move_right", m == 2)
	_press("action_1", int(action["jump"]) == 1)

func _press(a: String, on: bool) -> void:
	var name := "p%d_%s" % [slot, a]
	if on: Input.action_press(name)
	else: Input.action_release(name)

func get_obs() -> Dictionary:
	return {"obs": _features()}   # see 3.3

func get_reward() -> float:
	return reward
```

### 3.3 Observations

Start with a **vector** (works with `--headless`, 10–50× faster than pixels):

- P1 position relative to the flag, velocity, `is_on_floor()`
- the 3 bots' positions relative to P1
- the N nearest hazards: relative rect + velocity (reuse `PartyBot._dangers`)
- distance to the flag from the distance field (`PartyBot._field`, built per arena)
- phase one-hot (PICK / PLACE / COUNTDOWN / RUN / SCORE)

Later, for a pure-pixels Dreamer: add a `RGBCameraSensor2D` from the plugin (64×64),
exposed as obs key `image`. That needs a real window (no `--headless`).

### 3.4 Rewards (set `reward` in `DreamerController`, read by `get_reward()`)

| event | reward | hook |
|---|---|---|
| closer to the flag (distance-field cells) | `+0.1 × Δcells` | each physics frame in RUN |
| reached the flag | `+10` | `party_controller.gd:862` (`finished.append(slot)`) |
| first to the flag | `+5` | `finished[0] == 1` |
| eliminated (trap, pit) | `-5` | `game.eliminate_player(p)` / meta `eliminated` |
| points from own traps hitting bots | `+2` each | scoring in `_start_score()` (`POINTS_TRAP`) |
| time | `-0.01` / step | |

Using the same numbers as the game's scoring keeps "reward" ≈ "Party points".

### 3.5 Episodes and the non-race phases

One **episode = one round** (PICK → SCORE). `done = true; needs_reset = true` when
the phase becomes `SCORE`. Two options for PICK/PLACE:

- **A (start here):** a `PartyBot` handles pick and place for slot 1; the
  `DreamerController` only takes over in `RUN` (actions ignored and reward 0 before).
- **B (later):** Dreamer also picks and places (more actions, longer credit
  assignment — but that's where the trap points are).

### 3.6 Wiring

- Add a `Sync` node (from the plugin) to the party arena when the game is started with
  a new command-line option, e.g. `--rl` (add it in `core/command_line.gd`).
- In `party_controller.gd` after the bots are created (line ~78), if `--rl` is set,
  add `DreamerController` for slot 1 and `init(player1)`.
- `Sync` options: `action_repeat` (2–4), `speed_up` (e.g. 8) — both also settable from
  Python (`GodotEnv(action_repeat=..., speedup=...)`).

### 3.7 Test it without Dreamer

```powershell
cd C:/git/bot-party-dreamer
python -c "from godot_rl.core.godot_env import GodotEnv; e=GodotEnv(env_path=None); print(e.observation_space, e.action_space)"
```

`env_path=None` → "press PLAY in the Godot editor" with the `--rl` options in
**Project → Project Settings → Run → Main Run Args**. Then run a random policy for a
few rounds and check the rewards look right.

---

## 4. Export a training build

Training must not depend on the editor. Use `export_presets.cfg` (Windows Desktop):

```powershell
godot --headless --path C:/git/bot-party --export-release "Windows Desktop" C:/git/bot-party-dreamer/build/BotParty.exe
```

`GodotEnv(env_path="build/BotParty.exe", show_window=False)` launches it with
`--port`, `--action_repeat`, `--speedup` and any extra kwargs as `--key=value`, all as
engine args (no `--` separator, see the gotcha in 5.1).

---

## 5. Dreamer side (`bot-party-dreamer`)

### 5.1 `envs/botparty.py` — Dreamer env on top of `GodotEnv`

Same shape as `dreamerv3-torch/envs/crafter.py`:

```python
import gym
import numpy as np
from godot_rl.core.godot_env import GodotEnv

MOVES, JUMPS = 3, 2

class BotParty:
    metadata = {}

    def __init__(self, task, exe, port=11008, seed=0, action_repeat=2, speedup=8):
        # extra kwargs become "--key=value" engine args (see 5.1 note)
        self._env = GodotEnv(env_path=exe, port=port, seed=seed, show_window=False,
                             action_repeat=action_repeat, speedup=speedup,
                             rl="1", party="1", arena=task, players="1", bots="3")
        self._obs_dim = self._env.observation_space["obs"].shape[0]
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        b = lambda shape, dt=np.float32: gym.spaces.Box(-np.inf, np.inf, shape, dtype=dt)
        return gym.spaces.Dict({"state": b((self._obs_dim,)), "is_first": b((1,), np.uint8),
                                "is_last": b((1,), np.uint8), "is_terminal": b((1,), np.uint8)})

    @property
    def action_space(self):
        space = gym.spaces.Discrete(MOVES * JUMPS)
        space.discrete = True
        return space

    def _obs(self, raw, first=False, last=False, terminal=False):
        return {"state": np.asarray(raw[0]["obs"], np.float32), "is_first": first,
                "is_last": last, "is_terminal": terminal}

    def step(self, action):
        a = int(action) if np.isscalar(action) else int(np.argmax(action))
        obs, rew, done, trunc, info = self._env.step([[a % MOVES, a // MOVES]])
        done, trunc = bool(done[0]), bool(trunc[0])
        return self._obs(obs, last=done or trunc, terminal=done), np.float32(rew[0]), done or trunc, info[0]

    def reset(self):
        obs, _ = self._env.reset()
        return self._obs(obs, first=True)

    def close(self):
        self._env.close()
```

> **Gotcha:** `GodotEnv._launch_env` launches
> `BotParty.exe --port=11008 --env_seed=0 [--headless --disable-render-loop] --action_repeat=2 --speedup=8 --rl=1 --party=1 ...`
> — every option is an **engine** arg (no `--` separator). Bot Party's `CommandLine`
> only reads `OS.get_cmdline_user_args()` (after `--`), so in the `dreamer` branch make
> `core/command_line.gd` also parse `OS.get_cmdline_args()` (e.g. merge both when `rl`
> is present). The plugin's `Sync` node already reads `--port`, `--action_repeat`,
> `--speedup` from the engine args.

### 5.2 Register it in `dreamerv3-torch/dreamer.py` → `make_env`

```python
elif suite == "botparty":
    import envs.botparty as botparty
    env = botparty.BotParty(task, config.exe, port=11008 + id, seed=config.seed + id,
                            action_repeat=config.action_repeat)
    env = wrappers.OneHotAction(env)
```

(Either copy `botparty.py` into `dreamerv3-torch/envs/`, or add this repo to `PYTHONPATH`.)

### 5.3 Config block in `dreamerv3-torch/configs.yaml`

```yaml
botparty:
  task: botparty_steps
  exe: C:/git/bot-party-dreamer/build/BotParty.exe
  steps: 1e6
  action_repeat: 2
  envs: 1
  parallel: False          # Windows: keep False at first
  train_ratio: 512
  time_limit: 3000
  encoder: {mlp_keys: 'state', cnn_keys: '$^'}
  decoder: {mlp_keys: 'state', cnn_keys: '$^'}
  actor: {dist: 'onehot', std: 'none'}
  imag_gradient: 'reinforce'
```

### 5.4 Train and monitor

```powershell
cd C:/git/dreamerv3-torch
python dreamer.py --configs botparty --task botparty_steps --logdir C:/git/bot-party-dreamer/logdir/steps
tensorboard --logdir C:/git/bot-party-dreamer/logdir
```

Watch `train_return`, `eval_return`, and how often P1 is in `finished` / first.

---

## 6. Curriculum

1. `--arena=steps`, `--bot-skill=easy`, Dreamer only races (option A).
2. Bump bot skill → `normal` → `hard`.
3. `--arena=random` (Steps, Pillar, Windy heights, Meteor field) so it generalises.
4. More envs: several game instances, ports `11008 + id`, `envs: 4`, `parallel: True`.
5. Option B: Dreamer also picks and places traps.
6. Pixels: `RGBCameraSensor2D`, `cnn_keys: 'image'`, no `--headless`.

## 7. Play against it

Run the exported game with a window, `--rl`, and an inference script that loads the
Dreamer checkpoint (`logdir/.../latest.pt`) and calls the actor on each step. Your son
joins as P2 on the keyboard instead of a bot (`--players=2 --bots=2`, Dreamer on P1).

## Notes

- Train **offline only** — never with `--host` / `--public`.
- `dreamerv3-torch` is flagged outdated by its author; [r2dreamer](https://github.com/NM512/r2dreamer)
  is ~5× faster with the same env interface — switch once the pipeline works.
- Exploits found by the agent are worth showing to the game's author (see
  `bot-party/tests/exploit_search.gd`).
