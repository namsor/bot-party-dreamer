# Implementation log

What was actually done, step by step, following [dreamer3_steps.md](dreamer3_steps.md).
Each step says **what**, **why**, the **commands / files**, and the **result**.

---

## Step 0 — Environment inventory (2026-10-24)

| Item | Found | Consequence |
|---|---|---|
| Godot | `C:\Tools\Godot\Godot_v4.7.2-stable_win64_console.exe` (4.7.2) | matches `config/features = "4.7"`; the `_console` exe prints logs to the terminal, use it for scripts |
| Python | 3.13.3 only (`py -0`) | `dreamerv3-torch/requirements.txt` pins torch 2.4.1 / numpy 1.23.5 (no 3.13 wheels) → we install a **minimal** dependency set instead (step 2) |
| GPU | Intel Iris Xe, no NVIDIA / CUDA | Dreamer trains on **CPU** → vector observations, small model, short runs first |
| bot-party | branch `main` @ `1eeaa7b`, uncommitted `project.godot` (editor re-formatting only) | left as is, never committed by us; our work goes on branch `dreamer` |
| godot_rl_agents | @ `207b6f4`, plugin submodule **empty** | fetched over HTTPS (step 2) |
| dreamerv3-torch | clean clone | `dreamer.py` only needs torch, numpy, gym (spaces), ruamel.yaml, tensorboard |

Scripts in this repo find Godot through the `GODOT` environment variable (else `godot` on the PATH), e.g.:

```powershell
$env:GODOT = "C:\Tools\Godot\Godot_v4.7.2-stable_win64_console.exe"
```

---

## Step 1 — Game setup: 1 player (Dreamer) + 3 bots

**What:** start Party mode with only P1 joined and 3 computer opponents, without the menus.

**How it works in the game code**
- `--players=1` → `GameState.joined = [1]` (`core/command_line.gd`).
- `--bots=3 --bot-skill=easy` → `Party.bot_count = 3`, `Party.bot_skill = 0`.
- `--party` → `Party.new_game()` → `new_session()` gives the bots the free slots after the
  people (`party/party.gd:205`), so bots = **P2, P3, P4**.
- `PartyController._ready()` creates one `PartyBot` per bot slot (`party_controller.gd:78`).
- In the menus this is the same as: **Players** → join P1 only → bots row = 3.

**Check** — `tools/probe_party.gd` (in this repo) runs next to the game and prints the roster:

```powershell
& $env:GODOT --headless --path C:/git/bot-party --script C:/git/bot-party-dreamer/tools/probe_party.gd `
  -- --party --arena=steps --players=1 --bots=3 --bot-skill=easy
```

**Result**

```
Starting a party with 1 player(s)
t=5s  joined=[1] bots={ 2: 0, 3: 0, 4: 0 } phase=0 players=[]
...
t=30s joined=[1] bots={ 2: 0, 3: 0, 4: 0 } phase=0 players=[]
```

- ✅ P1 = person slot (future Dreamer), P2–P4 = Easy bots.
- ⚠️ The round stays in **PICK** (phase 0) forever: nobody picks an item for P1.
  → step 3 gives slot 1 a helper `PartyBot` for PICK / PLACE; Dreamer drives the RUN.
- ⚠️ `Party.new_game()` saves to `user://party_session.json` → it would **overwrite the
  "Continue" party** of whoever plays on this PC. Training processes get their own
  `APPDATA` (Godot's `user://` is `%APPDATA%\Godot\app_userdata\Bot Party`), see step 5.

---

## Step 2 — godot_rl_agents (local clone) + Python environment

### 2.1 Godot plugin

```powershell
cd C:/git/godot_rl_agents
git config submodule.godot_rl_agents_plugin.url https://github.com/edbeeching/godot_rl_agents_plugin.git
git submodule update --init
cd godot_rl_agents_plugin; git checkout main; git pull
```

- The submodule URL is SSH → switched to HTTPS (local git config only).
- ⚠️ The parent repo pins plugin commit `3984fd1` (**2023-01**, no `AIController2D`, old
  protocol). The Python package is protocol **0.7** → we use plugin `main` @ `998c357`
  (2026-06-13).

What the plugin gives us (and how it behaves — this drives the Godot design in step 3):

| file | role |
|---|---|
| `sync.gd` (`class_name Sync`) | TCP client to Python (port `--port`, default 11008). Every `action_repeat` physics frames: **pauses the tree**, sends `{obs, reward, done, info}`, waits for an action, unpauses. Collects agents (group `AGENT`) **once**, at startup. Reads `--speedup`, `--action_repeat`, `--env_seed` from engine args. |
| `controller/ai_controller_2d.gd` (`class_name AIController2D`) | base class: `get_obs`, `get_reward`, `get_action_space`, `set_action`, `get_info`, flags `done` / `needs_reset`. |
| `onnx/wrapper/ONNX_wrapper.gd` | referenced by `sync.gd`; loads a **C#** file → only works on Godot .NET. Copied for parsing only, never instantiated (we don't use ONNX inference). |

We copy **only** these three GDScript files into the game (step 3) and do **not** enable
the editor plugin: `class_name` works without it, so `project.godot` stays untouched.

### 2.2 Python virtualenv (`C:/git/bot-party-dreamer/.venv`)

Only Python 3.13 is installed, so `dreamerv3-torch/requirements.txt` (torch 2.4.1,
numpy 1.23.5, mujoco, dm_control…) can't be used as-is. Minimal set instead:

```powershell
cd C:/git/bot-party-dreamer
py -3.13 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install numpy "gymnasium<=1.0.0" "gym==0.26.2" ruamel.yaml tensorboard
.venv\Scripts\pip install --no-deps -e C:/git/godot_rl_agents
```

- `--no-deps` for godot_rl: skips stable-baselines3 / onnx / huggingface — we only use
  `godot_rl.core.godot_env.GodotEnv`.
- CPU torch (no NVIDIA GPU).
- Pinned in `requirements.txt` of this repo.

**Result:** `torch 2.14.1+cpu, numpy 2.5.4, gym 0.26.2, gymnasium 1.0.0`, `GodotEnv` imports.
`gym` warns about NumPy 2 — Dreamer only uses `gym.spaces`, checked in step 5.

### Python ↔ Godot protocol facts (from `godot_rl/core/godot_env.py`)
- `step()` returns `done` **twice** (no terminal / truncated split) → we pass
  `terminal` in the `info` dict from Godot.
- Extra `GodotEnv(**kwargs)` → `--key=value` **engine** args (before any `--`).
- On Windows `env_path` must end in `.exe`; `show_window=False` adds `--headless --disable-render-loop`.

---

## Step 3 — Godot side: Dreamer drives P1 (`bot-party`, branch `dreamer`)

**What:** let Python (godot_rl_agents) play P1 in a party against the 3 bots, one race
per episode, without changing how the game plays.

**Branch:** `git switch -c dreamer` in `C:/git/bot-party` (son's `main` untouched; the
pre-existing `project.godot` edit is left alone, not ours).

### 3.1 Design — why it looks like this

| Constraint (found by reading the code) | Decision |
|---|---|
| `Sync` collects its agents once; Bot Party **changes scene** for every party | `Sync` + controller live in a persistent node under `/root` (`RlTrainer`), not in the arena |
| `Sync` **pauses the tree** while waiting for Python | `RlTrainer.process_mode = ALWAYS` (Sync and the controller inherit it) |
| `Sync` steps every `action_repeat` frames, whatever the phase | `RlSync` (subclass) only steps while P1 races → PICK / PLACE / SCORE run free, no wasted Dreamer steps |
| Nobody picks / places for P1 → the round never starts (step 1) | `HelperBot extends PartyBot` does P1's pick & place (Hard skill), skips the race |
| `PartyBot` already has a distance map to the flag and a trap finder | `HelperBot.distance()` / `.dangers()` reuse them for reward and observations |
| Hazards are `Area2D` on layer 0 (rays can't see them) | traps come from `PartyBot._find_dangers()`; rays only for walls / floors / crates |
| `AIController2D` has a member var `_player` | our accessor is `_p1()` |
| ONNX wrapper needs C# | copied for parsing only; plugin not enabled → `project.godot` unchanged |

### 3.2 Files

```
bot-party/ (branch dreamer)
├── addons/godot_rl_agents/        subset of the plugin @ 998c357 (MIT): sync.gd,
│                                  controller/ai_controller_2d.gd, onnx/wrapper/ONNX_wrapper.gd
├── core/command_line.gd           +10 lines: --rl (and engine args when --rl is there)
└── rl/
    ├── rl_trainer.gd    RlTrainer        /root node: Sync + Dreamer, HelperBot per party,
    │                                     new party 3 s after one ends
    ├── rl_sync.gd       RlSync           Sync gated by DreamerController.tick()
    ├── dreamer_controller.gd DreamerController  keys, observations, rewards, episodes
    └── helper_bot.gd    HelperBot        pick & place for P1, distance map, trap finder
```

`core/command_line.gd` — godot_rl_agents passes `--port=… --speedup=…` as **engine** args;
with `--rl` they are merged into the options, and `--rl` adds `res://rl/rl_trainer.gd`
to `/root` before the party starts.

### 3.3 The agent (`DreamerController`)

**Actions** (godot_rl dict, discrete): `move` ∈ {none, left, right} × `jump` ∈ {released, held}
→ `Input.action_press/release("p1_move_left" | "p1_move_right" | "p1_action_1")`.

**Observation** — 65 floats (≈ -1..1, scale 1000 px):

| # | content |
|---|---|
| 7 | to the flag (x, y), velocity (x, y), on floor, distance-map cells / 100, race time left |
| 16 | rays every 22.5°, 480 px, walls/floors/crates: free fraction |
| 36 | 6 nearest traps: rect center offset (x, y), size (w, h), velocity (x, y) |
| 6 | 3 nearest other robots: offset (x, y) |

**Reward** (per physics frame, summed over `action_repeat`):

| event | reward |
|---|---|
| closer to the flag (distance-map cells) | `+0.1 × Δcells`, clamped ±1 per frame |
| time | `-0.002` per frame (≈ -0.12 / s) |
| reached the flag | `+10` (`+5` more if first) |
| eliminated (trap, pit, time out) | `-5` |
| race cut short (console / mode) | `0`, not terminal |

**Episode:** starts at "Go!" (phase RUN, P1 alive) → ends when P1 finishes / is eliminated
/ the race ends. `info = {terminal, outcome, episode}` (godot_rl has no terminal/truncated split).

**Protocol timeline**

```
RUN starts ─ tick()=true ─► Sync sends reset obs ◄─ Python reset() was waiting
   … step: obs, reward ◄─► action (every action_repeat frames, tree paused meanwhile)
P1 at flag / out ─ done=true sent ─► Python reset() → needs_reset → tick()=false
SCORE, PICK, PLACE, COUNTDOWN run free (HelperBot plays P1's pick & place)
next RUN ─ Sync sends the reset obs ─► next episode
party over (50 pts) → menu → RlTrainer starts a new party
```

### 3.4 Python launcher + smoke test (`bot-party-dreamer`)

- `botparty_dreamer/game.py` — `launch(port, arena, bots, bot_skill, headless, speedup,
  action_repeat, seed, exe)` starts Godot with the engine args for Sync and
  `-- --rl --party --arena=… --players=1 --bots=3 --bot-skill=…`, with
  **`APPDATA=runs/appdata/<port>`** (isolated saves).
- `tools/smoke_env.py` — `GodotEnv(env_path=None)` server + random actions.

```powershell
& $env:GODOT --headless --path C:/git/bot-party --import     # once: registers the new class_name scripts
cd C:/git/bot-party-dreamer
.venv\Scripts\python tools/smoke_env.py --episodes 3          # --window to watch
```

**Problems met and fixed:** `_player` name clash with `AIController2D`; two `:=`
type-inference errors on untyped `Node` (explicit types / `as CharacterBody2D`).

**Result** (speedup 8, action_repeat 4):

```
observation space [{'obs': {'size': [65], 'space': 'box'}}]
action space [{'move': {'size': 3, ...discrete}, 'jump': {'size': 2, ...discrete}}]
episode 0: first obs size=65 ... (waited 1.0s)
  steps=81  return=-11.75 info={'terminal': True, 'outcome': 'eliminated', 'episode': 1}
episode 1: ... (waited 1.4s)
  steps=363 return=-14.51 info={'terminal': True, 'outcome': 'eliminated', 'episode': 2}
episode 2: ... (waited 3.7s)
  steps=283 return=-4.56  info={'terminal': True, 'outcome': 'eliminated', 'episode': 3}
```

- ✅ handshake, spaces, 3 consecutive races, pick/place/score between them handled by the game.
- ✅ real `%APPDATA%\Godot\app_userdata\Bot Party\party_session.json` untouched.
- ℹ️ random play always gets eliminated (falling in pits); falling also costs progress
  reward (distance grows) on top of the -5.
- ℹ️ godot_rl reports the space as `Box(-1, 1)` but values can exceed 1 (e.g. 2.08):
  the Dreamer wrapper declares its own unbounded space (step 5).

---

## Step 4 — Training build: skipped (optional), editor binary used

**What was planned:** export `BotParty.exe` (preset **"Windows"**, `export_presets.cfg`) so
training doesn't need the editor binary.

**Found:** no export templates installed (`%APPDATA%\Godot\export_templates\` missing; they
are a ~1 GB download). Step 3 showed the editor binary works fine headless:
`Godot_v4.7.2-stable_win64_console.exe --headless --path C:/git/bot-party …`.

**Decision:** keep using the editor binary (`botparty_dreamer/game.py`, `exe=None`).
The only cost is a slightly slower start (imports are cached in `bot-party/.godot/`).

**To export later:**
1. Godot editor → **Editor → Manage Export Templates → Download and Install** (4.7.2).
2. Export from the branch `dreamer` (the `rl/` and `addons/` folders aren't excluded by the preset):
   ```powershell
   & $env:GODOT --headless --path C:/git/bot-party --export-release "Windows" C:/git/bot-party-dreamer/build/BotParty.exe
   ```
3. Train with `exe="C:/git/bot-party-dreamer/build/BotParty.exe"` (config key `exe`, step 5).

---

## Step 5 — Dreamer side: env wrapper, training script, ways to watch

**What:** plug the game into `dreamerv3-torch` **without modifying it**, and make training
visible (game window, TensorBoard curves, minimap videos, watch script).

### 5.1 Files (this repo)

| file | role |
|---|---|
| `botparty_dreamer/env.py` | `BotParty` Dreamer env on top of `GodotEnv` (same shape as `envs/crafter.py`) + `draw_minimap()` |
| `configs/botparty.yaml` | presets `botparty` (real run) and `smoke` (quick check), merged over dreamerv3-torch `defaults` |
| `train.py` | loads configs, swaps `dreamer.make_env`, applies 2 fixes, calls `dreamer.main()` unchanged |
| `watch.py` | loads `latest.pt`, races in a window at a chosen speed, prints first / finished |
| `tools/smoke_env.py` | random player, now with `--window --speedup N` |
| `README.md` | quick start |

### 5.2 The env (`BotParty`)

- **Action:** `Discrete(6)` = move {none, left, right} × jump {no, yes}; Dreamer gives a
  one-hot (`OneHotAction` wrapper) → `GodotEnv.step([[move], [jump]])`.
- **Observation dict:**
  - `state` (65 floats) — what the model learns from (`encoder/decoder mlp_keys: 'state'`);
  - `image` — required by dreamerv3-torch (`models.py:182` divides it by 255 even when no CNN
    uses it). Train env: 1×1 zeros (no memory cost). Eval env: **64×64 minimap** drawn from the
    state → TensorBoard `eval_policy` videos of *what Dreamer perceives*;
  - `is_first / is_last / is_terminal` — terminal from Godot's `info.terminal` (finished /
    eliminated); a race cut short is `is_last` only;
  - `log_<mode>_first / _finished / _eliminated` — 1 on the last step → Dreamer sums `log_*`
    keys per episode and logs them (the encoder ignores `log_*`).
- **Ports:** train envs `11008+i`, eval envs `11108+i`, watch `11208`.
- **Eval window:** the eval game is started with `--headless` off and `eval_speedup` (default 2),
  and **closes itself after `eval_episode_num` races** (Dreamer's eval loop never calls reset at
  the end, so the window would otherwise stay open and frozen during training). It reopens at
  the next evaluation.

Minimap legend: P1 white (center) + green velocity line, walls grey (ray hits), traps red,
bots blue, flag yellow (on the border when far).

### 5.3 Config (`configs/botparty.yaml`, preset `botparty`)

`device: cpu`, `compile: False`, `action_repeat: 4`, `envs: 1`, smaller world model
(`dyn_hidden/dyn_deter/units: 256`, MLP 3×256), `actor.dist: onehot`, `imag_gradient: reinforce`,
`time_limit: 4000` frames, `train_ratio: 128`, `eval_every: 2e4`, `eval_episode_num: 3`,
`video_pred_log: False`; game keys: `bots: 3`, `bot_skill: easy`, `speedup: 8`,
`eval_window: True`, `eval_speedup: 2`, `exe: ''`. Any key can be overridden: `--bot_skill hard`.

`smoke` preset: `steps 4000, prefill 400, eval_every 2000, eval_episode_num 1, batch 8×32`.

### 5.4 Problems met and fixed

| problem | fix |
|---|---|
| `dreamer.py __main__` uses `yaml.safe_load`, removed from current ruamel.yaml | `train.py` loads configs with `YAML(typ="safe")` and reuses `dreamer.main()` |
| gym 0.26: `ValueError: Unknown dtype uint8 for infinite bounds` | flag boxes bounded 0..1 |
| `AssertionError: Torch not compiled with CUDA enabled` — `networks.MLP` and `tools.DiscDist` default to `device="cuda"` and the code never overrides it | `train.use_device()` rewrites those defaults to `config.device` |
| TensorBoard video: `PermissionError ...\Temp\tmp*.gif` (torch reopens an open `NamedTemporaryFile`, forbidden on Windows) | `train.gif_videos()` replaces torch's `make_video` with an in-memory Pillow GIF (×4 zoom), also saved as `logdir/<run>/videos/eval_NNNN.gif` |
| torch's `add_video` needs `moviepy.editor` (moviepy 1.x) | `moviepy==1.0.3` (in `requirements.txt`) |
| failed runs left Godot processes behind | killed by hand; normal exits close them (`env.close()`) |
| `torch.cuda.amp.autocast` FutureWarnings | harmless (AMP is off at `precision: 32`) |

### 5.5 Smoke run — result

```powershell
.venv\Scripts\python train.py --configs botparty smoke
```

Exit 0 after ~45 min (19:46 → 20:30), outputs in `logdir/smoke/`: `latest.pt` (47 MB),
`metrics.jsonl`, TensorBoard events, `train_eps/`, `eval_eps/`, `videos/eval_000{0,1,2}.gif`.

| frames | what |
|---|---|
| 0 | prefill (random): 2 races, returns -11.6, -4.4, eliminated (falls) |
| 1600 | eval 1 (window, speed 2): return -9.6, 599 steps = timeout, eliminated |
| 1600 → 7200 | 800 updates: `model_loss 13.0 → 3.6`, `state_loss 7.5 → 1.2`, `reward_loss 4.4 → 1.6` |
| 3600 | eval 2: -4.9, 80 steps (fell) |
| 5600 | eval 3: -9.3, 599 steps (timeout) |

- ✅ **The whole pipeline works**: game ↔ godot_rl ↔ Dreamer, prefill, world-model + actor-critic
  updates, evaluation in a window, per-race stats, minimap videos, checkpoint.
- ✅ The world model learns (losses drop 3–6×). Too short to expect a better racer.
- ⚠️ **Speed: ~2.5 game frames / s** (`fps` in `metrics.jsonl`) ≈ one update every ~3 s on CPU
  (12 threads). The game is not the bottleneck (~80 env steps / s alone, step 3); the
  updates are. At this rate the default `steps: 1e6` would take **~4.5 days** → step 6.

### 5.6 Watching training — the three ways

1. **Game window:** eval races open a window at `eval_speedup` (default 2) every `eval_every`
   frames, then close; or `tools/smoke_env.py --window --speedup 1` for a random player.
2. **TensorBoard** (a viewer, training itself runs on CPU):
   `.venv\Scripts\tensorboard --logdir logdir` → http://localhost:6006.
   Scalars: `train_return`, `eval_return`, `log_train_first/finished/eliminated`, losses.
   Images: `eval_policy` (minimap GIF). Same GIFs in `logdir/<run>/videos/`.
3. **`watch.py`** — a checkpoint racing in a window:
   `.venv\Scripts\python watch.py --logdir logdir/smoke --speedup 1 --races 3`.

**`watch.py` check** (smoke checkpoint, 1 race, window, speed 4):

```
Optimizer model_opt has 2863681 variables. / actor_opt 395782 / value_opt 459775
Loaded logdir\smoke\latest.pt
race 1: eliminated return   -9.61
first 0/1, finished 0/1
```
✅ loads, races in a window, closes the game.

---

## Step 6 — Short real training run (started 2026-10-24 20:37, resumed from smoke)

**Choice:** short test (~2–3 h) rather than overnight / multi-day: check that the curves move
in the right direction before committing the PC for longer.

**Preset `short`** (`configs/botparty.yaml`, on top of `botparty`):
`steps 8e4` frames (≈ 20k Dreamer steps, a few hundred races), `train_ratio 32`
(4× fewer updates per step than `botparty`, ≈ one update per 32 steps), `eval_every 2e4`
(4 evaluations), `eval_episode_num 3` (3 races in a window at speed 2 each time),
`logdir logdir/short`. Opponents: 3 **Easy** bots on the **Steps** arena.

**Resumed from the smoke run** (instead of starting from scratch): Dreamer resumes when the
logdir already holds `latest.pt` (model + optimizers) and `train_eps/` (recorded races). Same
model sizes, only the batch differed (8×32 → 16×64), which doesn't change the weights.

```powershell
Copy-Item -Recurse logdir\smoke logdir\short          # then delete events.out.tfevents.* and metrics.jsonl
.venv\Scripts\python -u train.py --configs botparty short --prefill 0 > runs/train_short.out
.venv\Scripts\tensorboard --logdir logdir       # http://localhost:6006, run "short"
```

**What to watch:** `log_train_finished` / `log_train_first` leaving 0, `train_return` rising
above the random level (≈ -10), `eval_return`; minimap GIFs in `logdir/short/videos/`.

Start check: `Prefill dataset (0 steps)`, `Logger: (5232 steps)` → continues the smoke run
(TensorBoard events not copied: the `short` curves start clean, `smoke` stays separate).

**Evaluation every 10 minutes** (asked: "a game window more often, every 10 minutes").
Dreamer only knows `eval_every` in frames, so `train.py` got `eval_every_minutes()`: it wraps
`tools.simulate` and cuts each training call (`steps=eval_every`) into 50-step chunks, returning
once 10 min have passed. `dreamer.main()` then saves `latest.pt` and runs the evaluation (window,
3 races at speed 2) as usual. Config key `eval_minutes` (`botparty`: 0 = off, `short`: 10).
Side effect: a checkpoint every ~10 min, so a stopped run loses at most 10 min of updates.
Restarted at 20:42 (from frame 7824: the aborted run's 2 races were kept in `train_eps/`,
its updates were lost — no checkpoint yet). First eval: 3 races, `eval_return -9.9`, all timeouts.

**Result** — stopped at frame ~63,400 of 80,000 (≈ 22:10): Claude Code killed the run and
TensorBoard because the PC ran critically low on memory (Python ≈ 1 GB + Godot + TensorBoard
+ eval window). Last checkpoint `logdir/short/latest.pt` at 22:09:46 → resumable with the same command.

| frames | train races | eval |
|---|---|---|
| 49k–51k | eliminated after 90–200 steps (falls), return -11 … -13 | -11.4, 54 steps |
| 53k–63k | **survives the full 40 s** (599–600 steps), return -8.7 … -9.9 | -9.6 / -9.9, 599 steps |

- ❌ never reached the flag (`log_*_finished` = 0 in every race).
- ✅ it learned something: **stop falling into pits** — races went from ~100 steps to the full
  clock, `actor_entropy` 1.5 → 0.8 (the policy became more decided).
- ⚠️ that is a **local optimum** of the reward: the time-out costs the same -5 as a pit, and
  progress (+0.1 per cell) is too weak to pull it to the flag. Next fixes to try: a larger progress
  reward / a bonus for the best distance reached, a smaller penalty for timing out than for dying,
  then a longer run.
- Speed: `fps 8.5` frames/s at `train_ratio 32` (vs 2.5 at 128 in the smoke run).


---

## Step 7 — Published on GitHub (2026-10-24)

`https://github.com/namsor/bot-party-dreamer` (public, MIT), on top of its "Initial commit"
(`.gitignore` = GitHub Python template, + `logdir/`, `runs/`).

- **Godot side copied in:** `godot/rl/*.gd`, `godot/addons/godot_rl_agents/` (plugin subset,
  MIT + LICENSE), `godot/command_line.patch` (diff of `core/command_line.gd` vs bot-party
  `main` @ `1eeaa7b`). `tools/install_godot.py <bot-party>` switches the checkout to branch
  `dreamer`, copies the files, applies the patch (idempotent). Tested on a fresh clone of
  bot-party `main`: import without script errors, the 4 classes registered, files identical to
  the working install.
- **Personal paths removed:** `GODOT` env var (else `godot` on the PATH, clear error otherwise),
  `BOT_PARTY` / `DREAMER` default to sibling folders `../bot-party`, `../dreamerv3-torch`.
  Docs use `C:\Tools\Godot\...` as an example path.
- **Not published:** `logdir/` (checkpoints, recorded races, TensorBoard), `runs/` (game logs,
  isolated APPDATA), `.venv/`. One eval minimap GIF (`doc/img/eval_minimap.gif`) for the README.

---

## Step 8 — Reward adjustment, then resume the short run (2026-10-24 22:23)

**Why:** the short run settled in a local optimum, "survive until the clock runs out": a
time-out cost the same -5 as a pit, falling also cost progress reward (the distance grows while
falling), and +0.1 per cell was too weak to pay for the risk of moving. Dreamer had never seen
the flag, so the +10 / +5 didn't exist for its world model.

**New rewards** (`godot/rl/dreamer_controller.gd`):

| | before | after |
|---|---|---|
| progress | +0.1 × Δ distance, ±1 / frame (negative when moving away / falling) | **+0.3 per cell closer than ever before in the race** (max 1 / frame): no penalty for stepping back or falling, nothing to farm |
| time | -0.002 / frame | **-0.001 / frame** |
| clock runs out | -5 (same as a pit) | **-1** (outcome `timeout`, detected as `run_time ≥ limit` when eliminated) |
| trap / pit | -5 | -5 |
| flag / first | +10 / +5 | +10 / +5 |

Rough value of a 40 s race: flag ≈ +30 ≫ 15 cells then a fall ≈ -1 > wait for the clock ≈ -3.4 >
early fall ≈ -4 > immediate death ≈ -5. (With the old time penalty, dying at once would have
beaten waiting — checked before choosing -0.001.)

**Recorded races relabeled** — `tools/relabel_rewards.py logdir/short` recomputes the rewards
of the saved races from the state (index 5 = distance cells / 100) and the `log_*` flags, and
adds `log_<mode>_timeout` (otherwise episodes with different keys would break Dreamer's sampler).
Originals kept in `train_eps.orig/`, `eval_eps.orig/`.

```
train_eps: 109 races {'eliminated': 100, 'timeout': 9}; mean return -9.13 -> 1.38
eval_eps:  33 races  {'timeout': 16, 'eliminated': 17}; mean return -9.97 -> 1.06
```

**Env:** new stat `log_<mode>_timeout` (TensorBoard), `eliminated` now means trap / pit only.

**Check:** `tools/smoke_env.py --episodes 3` (random): no script error, returns 0.81, -1.26, -0.52
(were ≈ -12).

**Resumed** from `latest.pt` (22:09) with the relabeled data:
`train.py --configs botparty short --prefill 0` → `Logger: (62652 steps)`; first eval (old policy,
new rewards): 3 × timeout, `eval_return -0.1`. ~17k frames left (≈ 35–40 min).

**Result** — the run completed (frame 81,052 ≥ 80,000; last checkpoint 23:37):

| frames | train races | eval |
|---|---|---|
| 62.6k | — | 3 × timeout, -0.1 |
| 65k–72k | all timeouts, return 0.5, -0.1, 0.2, 0.2 | 3 × timeout, -0.4 (72.4k) |
| 74k–79k | all timeouts, return 0.2, **1.1, 0.9** | 3 × timeout, -0.1 (81k) |

- ✅ no more falls: every race lasts the full clock.
- ↗ return creeps up (≈ 0 → ≈ 1): it covers a bit more new ground per race.
- ❌ still never reaches the flag — ~18k frames (≈ 75 min) is too short to judge the new rewards.

---

## Step 9 — Long training run, initialized from the short run (started 2026-10-24 23:41)

```powershell
mkdir logdir\steps; copy logdir\short\latest.pt, train_eps\, eval_eps\ into it   # no TensorBoard logs, videos, .orig backups
.venv\Scripts\python -u train.py --configs botparty --prefill 0 --eval_minutes 30 > runs/train_steps.out
```

- Preset `botparty`: `steps 1e6` frames, `train_ratio 128` (4× more updates per step than `short`),
  same model sizes as `short` → the weights load as is.
- `--eval_minutes 30`: a checkpoint and an evaluation window (3 races, speed 2) every ~30 min,
  instead of every 2e4 frames (≈ 2 h at this speed) — limits the loss if the run is stopped again.
- Resumed at `Logger: (79428 steps)` (the recorded races); first eval: 3 × timeout, `eval_return -0.7`.
- Expected speed ≈ 2.5 frames/s (smoke run, same `train_ratio`) → ~920k frames left ≈ **4–5 days**.
  Resume after any stop with the same command (`logdir/steps/latest.pt`).

**Result so far** (2026-10-25 07:35, frame 216k of 1M, **4.4 frames/s** → ~2 more days; python 2.6 GB RAM):

Races per hour, from the recorded races (`train_eps`, state index 5 = distance-map cells; a race
starts ≈ 130 cells from the flag; "way" = share of that distance covered at best):

| hour | races | mean return | mean way | best way | flag reached |
|---|---|---|---|---|---|
| before (smoke + short) | 117 | 1.3 | 17% | 46% | 0 |
| 00h–02h | 276 | ≈ 0.5 | 13–17% | 73% | 0 |
| 03h | 52 | 7.6 | 37% | 85% | 0 |
| 04h | 78 | 9.1 | 37% | 99% | 3 (2 first) |
| 05h | 51 | 9.0 | 37% | 93% | 0 |
| 06h | 76 | 8.1 | 37% | 99% | 1 |
| 07h (40 min) | 40 | 11.9 | 44% | 99% | 3 (2 first), best return +45.1 |

- **~01:00, it stops waiting:** nearly every race ends in a trap or pit after ~50 steps (12 s)
  instead of timing out — the new rewards at work (step 8).
- **~03:00, it goes much further:** mean way 15% → 37%, then 44%.
- **From 04:00, it reaches the flag:** 7 times in ~300 races, 4 of them first. Only in training races
  (sampled actions); the evaluation races (best action) don't reach it yet but improved too:
  mean way 10–18% → 46–57%, best 87%.
- Next hurdle: getting past the traps reliably (36 of the 40 races of 07h ended in one).

**Next ideas (not done):** once the flag is reached in most races, continue from this checkpoint with
`--bot_skill hard` (finishing first becomes the challenge) and several arenas (racing in general,
not one memorized map). Self-play would only pay off for the item-placement phase, where players
really act on each other; it's done by `HelperBot` today.

