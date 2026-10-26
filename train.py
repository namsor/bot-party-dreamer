r"""Train DreamerV3 (dreamerv3-torch) on Bot Party.

    .venv\Scripts\python train.py --configs botparty
    .venv\Scripts\python train.py --configs botparty smoke          # quick check
    .venv\Scripts\python train.py --configs botparty --bot_skill normal --logdir logdir/normal

Config = dreamerv3-torch/configs.yaml "defaults" + the named presets (from that file or
configs/botparty.yaml) + --key value overrides. dreamer.main() is used as is; only
make_env is swapped for Bot Party.
"""

import argparse
import os
import pathlib
import sys
import time

from ruamel.yaml import YAML

ROOT = pathlib.Path(__file__).resolve().parent
DREAMER = pathlib.Path(os.environ.get("DREAMER", ROOT.parent / "dreamerv3-torch"))  # else ../dreamerv3-torch
sys.path.insert(0, str(DREAMER))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # logdir, runs/ are relative to this repo

import dreamer  # noqa: E402
import envs.wrappers as wrappers  # noqa: E402
import tools  # noqa: E402

from botparty_dreamer.env import BotParty  # noqa: E402


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    assert suite == "botparty", config.task
    train = "train" in mode
    env = BotParty(
        task,
        mode="train" if train else "eval",
        port=config.port + id + (0 if train else 100),
        seed=config.seed + id,
        action_repeat=config.action_repeat,
        speedup=config.speedup if train else config.eval_speedup,
        bots=config.bots,
        bot_skill=config.bot_skill,
        window=not train and config.eval_window,
        sound=config.eval_sound,
        minimap=not train,  # eval videos in TensorBoard
        episodes_per_launch=0 if train else max(1, config.eval_episode_num),
        exe=config.exe or None,
    )
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def use_device(device):
    """dreamerv3-torch has two device="cuda" defaults (networks.MLP, tools.DiscDist) that the
    code never overrides: point them to `device` (CPU here)."""
    import networks

    for fn in (networks.MLP.__init__, tools.DiscDist.__init__):
        fn.__defaults__ = tuple(device if d == "cuda" else d for d in fn.__defaults__)


def compile_networks(enabled):
    """dreamerv3-torch compiles only off Windows, and then only WorldModel/ImagBehavior.forward(),
    which training never calls (it calls _train()). Compile instead, in place, the MLPs and the
    GRU cell that each update runs thousands of times: ~20% faster per update on GPU (needs
    triton-windows on Windows). State dict keys are unchanged, so checkpoints still load."""
    if not enabled:
        return
    import networks
    import torch._dynamo.config
    import torch._functorch.config

    torch._functorch.config.donated_buffer = False  # dreamerv3-torch's backward(retain_graph=True)
    # All MLPs share one forward(): one compiled version per instance and input shape, more than 8
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.accumulated_recompile_limit = 1024
    original = dreamer.Dreamer.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for module in self.modules():
            if isinstance(module, (networks.MLP, networks.GRUCell)):
                module.compile()

    dreamer.Dreamer.__init__ = __init__


def load_new_parts():
    """Resume a run with parts its checkpoint doesn't have yet (e.g. turning on plan2explore), or
    that changed size: load what fits, keep the rest fresh. dreamer.main() loads strictly."""
    original = dreamer.Dreamer.load_state_dict

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        own = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k not in own or own[k].shape == v.shape}
        if self._expl_behavior is self._task_behavior:
            # Exploration off: _expl_behavior is the task actor-critic, don't load an old explorer
            # into it.
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_expl_behavior.")}
        result = original(self, state_dict, strict=False, **kwargs)
        # The explorer's copy of the world model is the same module as _wm: already loaded.
        new = sorted({k.split(".")[0] for k in result.missing_keys if "_world_model" not in k})
        if self._expl_behavior is self._task_behavior and "_expl_behavior" in new:
            new.remove("_expl_behavior")  # an alias of _task_behavior, loaded under that name
        if new:
            print(f"checkpoint: new parts start fresh: {', '.join(new)}", flush=True)
        return result

    dreamer.Dreamer.load_state_dict = load_state_dict

    # Same for the optimizers: skip the ones the agent doesn't have (e.g. plan2explore turned off).
    def load_optims(obj, optimizers_state_dicts):
        for path, state in optimizers_state_dicts.items():
            target = obj
            for key in path.split("."):
                target = getattr(target, key, None)
            if target is None:
                print(f"checkpoint: optimizer {path} no longer used", flush=True)
            else:
                target.load_state_dict(state)

    tools.recursively_load_optim_state_dict = load_optims


def save_best(logdir, races=100):
    """dreamerv3-torch only keeps latest.pt, overwritten at each evaluation. After each save, also
    copy it to best.pt when the last `races` training races reached the flag more often than ever
    (mean return breaks ties); the score is in best.json. Training races, not the 3 eval races:
    far less noise."""
    import json
    import shutil

    import torch

    logdir = pathlib.Path(logdir)
    original = torch.save

    def save(obj, f, *args, **kwargs):
        original(obj, f, *args, **kwargs)
        if not isinstance(f, (str, pathlib.Path)) or pathlib.Path(f).name != "latest.pt":
            return
        rows = []
        with open(logdir / "metrics.jsonl", encoding="utf-8") as m:
            for line in m:
                if '"log_train_finished"' in line:
                    rows.append(json.loads(line))
        rows = rows[-races:]
        if len(rows) < races:
            return
        score = {
            "step": rows[-1]["step"],
            "flag": sum(r["log_train_finished"] for r in rows) / races,
            "first": sum(r["log_train_first"] for r in rows) / races,
            "mean_return": sum(r["train_return"] for r in rows) / races,
            "races": races,
            "time": time.strftime("%Y-%m-%d %H:%M"),
        }
        best = logdir / "best.json"
        old = json.loads(best.read_text()) if best.exists() else None
        if old and (score["flag"], score["mean_return"]) <= (old["flag"], old["mean_return"]):
            return
        shutil.copyfile(f, logdir / "best.pt")
        best.write_text(json.dumps(score, indent=1))
        print(f"best.pt: last {races} training races reached the flag {100 * score['flag']:.0f}% "
              f"(first {100 * score['first']:.0f}%), mean return {score['mean_return']:.1f}", flush=True)

    torch.save = save


def gif_videos(logdir, zoom=4):
    """TensorBoard videos without temp files: torch's make_video reopens a NamedTemporaryFile,
    which Windows forbids (PermissionError). Encode the GIF in memory with Pillow instead, and
    also keep it as logdir/videos/eval_<n>.gif."""
    import io

    from PIL import Image
    from torch.utils.tensorboard import summary

    videos = pathlib.Path(logdir) / "videos"
    videos.mkdir(parents=True, exist_ok=True)

    def make_video(tensor, fps):
        _t, h, w, c = tensor.shape
        frames = [Image.fromarray(f).resize((w * zoom, h * zoom), Image.NEAREST) for f in tensor]
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
        data = buf.getvalue()
        (videos / f"eval_{len(list(videos.glob('*.gif'))):04d}.gif").write_bytes(data)
        return summary.Summary.Image(height=h * zoom, width=w * zoom, colorspace=c, encoded_image_string=data)

    summary.make_video = make_video


def eval_every_minutes(minutes, chunk=50):
    """Evaluate (and checkpoint) at least every `minutes` of wall time, not only every
    `eval_every` frames: dreamer.main() trains with tools.simulate(steps=eval_every), so a
    training call is cut into chunks and returns once the time is up. 0 = off."""
    original = tools.simulate

    def simulate(agent, envs, cache, directory, logger, is_eval=False, limit=None, steps=0, episodes=0,
                 state=None):
        if is_eval or not steps or not minutes or not isinstance(agent, dreamer.Dreamer):
            return original(agent, envs, cache, directory, logger, is_eval, limit, steps, episodes, state)
        end = time.time() + 60 * minutes
        done = 0
        while done < steps and time.time() < end:
            n = min(chunk, steps - done)
            state = original(agent, envs, cache, directory, logger, is_eval, limit, n, 0, state)
            done += n
        return state

    tools.simulate = simulate


def load_config(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["botparty"])
    args, remaining = parser.parse_known_args(argv)
    yaml = YAML(typ="safe", pure=True)
    configs = yaml.load((DREAMER / "configs.yaml").read_text())
    configs.update(yaml.load((ROOT / "configs" / "botparty.yaml").read_text()))

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    merged = {}
    for name in ["defaults", *args.configs]:
        recursive_update(merged, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(merged.items()):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining)


if __name__ == "__main__":
    config = load_config(sys.argv[1:])
    use_device(config.device)
    compile_networks(config.compile)
    load_new_parts()
    save_best(config.logdir)
    gif_videos(config.logdir)
    eval_every_minutes(config.eval_minutes)
    dreamer.make_env = make_env
    dreamer.main(config)
