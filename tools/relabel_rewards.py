r"""Recompute the rewards of recorded races after a reward change (DreamerController, Godot).

    .venv\Scripts\python tools/relabel_rewards.py logdir/short

Rewrites <logdir>/train_eps/*.npz and eval_eps/*.npz in place, after copying them to
<dir>.orig/ (once). Uses the state saved at every step (index 5 = distance to the flag in
cells / 100) and the per-race log_* flags. Mirrors godot/rl/dreamer_controller.gd:
  progress  +R_PROGRESS per cell closer to the flag than ever before in the race (max 1 / frame)
  time      R_TIME per frame
  end       finished +R_FINISH (+R_FIRST if first), timeout R_TIMEOUT, other elimination R_ELIMINATED
Also adds the log_<mode>_timeout key (a race that lasted the whole clock is a timeout).
"""

import argparse
import pathlib
import shutil

import numpy as np

R_FINISH, R_FIRST, R_ELIMINATED, R_TIMEOUT = 10.0, 5.0, -5.0, -1.0
R_PROGRESS, R_TIME = 0.3, -0.001
ACTION_REPEAT = 4
RACE_STEPS = 40 * 60 // ACTION_REPEAT  # 40 s race = 600 steps
DIST = 5  # state index: distance-map cells / 100 (1.0 = unknown)


def relabel(ep, mode):
    n = len(ep["reward"])
    dist = ep["state"][:, DIST].astype(np.float64) * 100.0
    known = ep["state"][:, DIST] != 1.0
    reward = np.zeros(n, np.float32)
    best = dist[0] if known[0] else np.inf
    for t in range(1, n):
        r = R_TIME * ACTION_REPEAT
        if known[t] and dist[t] < best:
            if np.isfinite(best):
                r += min(R_PROGRESS * (best - dist[t]), 1.0 * ACTION_REPEAT)
            best = dist[t]
        reward[t] = r
    first, finished, elim = (ep[f"log_{mode}_{k}"] for k in ("first", "finished", "eliminated"))
    timeout = np.zeros(n, np.float32)
    if ep["is_terminal"][-1]:
        if finished[-1]:
            reward[-1] += R_FINISH + (R_FIRST if first[-1] else 0.0)
        elif elim[-1] and n - 1 >= RACE_STEPS - 1:
            reward[-1] += R_TIMEOUT
            timeout[-1], elim = 1.0, elim.copy()
            elim[-1] = 0.0
        elif elim[-1]:
            reward[-1] += R_ELIMINATED
    out = dict(ep)
    out.update({"reward": reward, f"log_{mode}_eliminated": elim, f"log_{mode}_timeout": timeout})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logdir", type=pathlib.Path)
    args = p.parse_args()
    for sub, mode in (("train_eps", "train"), ("eval_eps", "eval")):
        folder = args.logdir / sub
        if not folder.exists():
            continue
        backup = folder.with_name(sub + ".orig")
        if not backup.exists():
            shutil.copytree(folder, backup)
        files = sorted(folder.glob("*.npz"))
        outcomes = {}
        old_total = new_total = 0.0
        for f in files:
            ep = dict(np.load(backup / f.name))
            new = relabel(ep, mode)
            old_total += ep["reward"].sum()
            new_total += new["reward"].sum()
            k = next((k for k in ("first", "finished", "timeout", "eliminated") if new[f"log_{mode}_{k}"][-1]), "cut")
            outcomes[k] = outcomes.get(k, 0) + 1
            np.savez_compressed(f, **new)
        if files:
            print(f"{sub}: {len(files)} races {outcomes}; mean return {old_total / len(files):.2f} -> "
                  f"{new_total / len(files):.2f} (backup: {backup.name}/)")


if __name__ == "__main__":
    main()
