r"""Watch a trained Dreamer race the bots, in a window.

    .venv\\Scripts\\python watch.py --logdir logdir/steps                  # latest.pt, speed 1
    .venv\\Scripts\\python watch.py --logdir logdir/steps --speedup 2 --races 5 --bot_skill hard

Uses the same config as training (--configs botparty ... and --key value overrides).
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

import train  # sets up paths (dreamerv3-torch) and chdir to this repo
from botparty_dreamer.env import BotParty

import dreamer  # noqa: E402
import envs.wrappers as wrappers  # noqa: E402
import tools  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--races", type=int, default=3)
parser.add_argument("--speedup", type=float, default=1.0, help="game speed (1 = real time)")
parser.add_argument("--checkpoint", default="", help="default: <logdir>/latest.pt")
args, remaining = parser.parse_known_args(sys.argv[1:])
config = train.load_config(remaining)
train.use_device(config.device)
logdir = pathlib.Path(config.logdir)
checkpoint = pathlib.Path(args.checkpoint or logdir / "latest.pt")

env = BotParty(config.task.split("_", 1)[1], mode="watch", port=config.port + 200, seed=config.seed,
               action_repeat=config.action_repeat, speedup=args.speedup, bots=config.bots,
               bot_skill=config.bot_skill, window=True, exe=config.exe or None)
env = wrappers.SelectAction(wrappers.OneHotAction(env), key="action")
config.num_actions = env.action_space.shape[0]
config.time_limit //= config.action_repeat

agent = dreamer.Dreamer(env.observation_space, env.action_space, config, tools.Logger(logdir / "watch", 0), None)
agent.load_state_dict(torch.load(checkpoint, map_location=config.device)["agent_state_dict"])
agent.requires_grad_(requires_grad=False)
agent.eval()
print(f"Loaded {checkpoint}")

wins = finishes = 0
try:
    for race in range(args.races):
        o, done, state, ret = env.reset(), True, None, 0.0
        while True:
            obs = {k: np.stack([v]) for k, v in o.items() if not k.startswith("log_")}
            out, state = agent(obs, np.array([done]), state, training=False)
            o, r, done, info = env.step({"action": out["action"][0].detach().cpu().numpy()})
            ret += r
            if done:
                break
        outcome = info.get("outcome", "")
        wins += outcome == "first"
        finishes += outcome in ("first", "finished")
        print(f"race {race + 1}: {outcome:10s} return {ret:7.2f}")
finally:
    env.close()
print(f"first {wins}/{args.races}, finished {finishes}/{args.races}")
