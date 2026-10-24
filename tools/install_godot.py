r"""Install the Godot side (godot/) into a Bot Party checkout.

    python tools/install_godot.py C:/git/bot-party

- switches the checkout to a branch "dreamer" (created from the current branch), so main stays clean;
- copies godot/rl/ and godot/addons/godot_rl_agents/ into it;
- applies godot/command_line.patch (the --rl option) unless it is already applied.
Then run once: godot --headless --path <checkout> --import   (registers the new class_name scripts)
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GODOT_DIR = ROOT / "godot"


def git(game, *args, check=True):
    return subprocess.run(["git", "-C", str(game), *args], capture_output=True, text=True, check=check)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("game", type=pathlib.Path, help="Bot Party checkout (folder with project.godot)")
    p.add_argument("--branch", default="dreamer", help="branch to install on ('' = stay on the current one)")
    args = p.parse_args()
    game = args.game.resolve()
    project = game / "project.godot"
    if not project.exists() or 'config/name="Bot Party"' not in project.read_text(encoding="utf-8"):
        sys.exit(f"{game} is not a Bot Party project")

    if args.branch:
        current = git(game, "branch", "--show-current").stdout.strip()
        if current != args.branch:
            exists = git(game, "rev-parse", "--verify", "--quiet", args.branch, check=False).returncode == 0
            git(game, "switch", *([] if exists else ["-c"]), args.branch)
            print(f"branch: {args.branch} ({'existing' if exists else 'new, from ' + current})")

    for sub in ("rl", "addons/godot_rl_agents"):
        shutil.copytree(GODOT_DIR / sub, game / sub, dirs_exist_ok=True)
        print(f"copied: {sub}/")

    patch = str(GODOT_DIR / "command_line.patch")
    if git(game, "apply", "--reverse", "--check", patch, check=False).returncode == 0:
        print("patch: core/command_line.gd already has --rl")
    else:
        result = git(game, "apply", patch, check=False)
        if result.returncode:
            sys.exit(f"patch failed (made for bot-party main @ 1eeaa7b):\n{result.stderr}")
        print("patch: core/command_line.gd now has --rl")
    print(f"next: godot --headless --path {game} --import")


if __name__ == "__main__":
    main()
