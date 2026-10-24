"""Start Bot Party in training mode (--rl), one process per port.

Each process gets its own APPDATA, so Godot's user:// (saves, settings) is
runs/appdata/<port>/Godot/app_userdata/Bot Party and never touches the real one.
"""

import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Godot 4.7 binary (on Windows the *_console.exe one): $GODOT, else "godot" on the PATH.
GODOT = os.environ.get("GODOT") or shutil.which("godot") or shutil.which("godot4")
# Bot Party checkout with the Godot side installed (tools/install_godot.py): $BOT_PARTY, else ../bot-party.
PROJECT = os.environ.get("BOT_PARTY", str(ROOT.parent / "bot-party"))


def launch(port=11008, arena="steps", bots=3, bot_skill="easy", headless=True, speedup=8,
           action_repeat=4, seed=0, exe=None, log=None):
    """Start the game; it connects to the godot_rl server listening on `port`.

    exe: an exported BotParty.exe (step 4); default runs the project with the Godot editor binary.
    """
    if not exe and not GODOT:
        raise RuntimeError("Godot not found: set GODOT to the Godot 4.7 binary (Windows: the *_console.exe)")
    cmd = [exe] if exe else [GODOT, "--path", PROJECT]
    if headless:
        cmd += ["--headless"]
    # Engine args read by the plugin's Sync node (and by bot-party's command line when --rl is there).
    cmd += [f"--port={port}", f"--speedup={speedup}", f"--action_repeat={action_repeat}", f"--env_seed={seed}", "--rl"]
    # Bot Party options (after "--").
    cmd += ["--", "--rl", "--party", f"--arena={arena}", "--players=1", f"--bots={bots}", f"--bot-skill={bot_skill}"]
    appdata = ROOT / "runs" / "appdata" / str(port)
    appdata.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, APPDATA=str(appdata))
    out = open(log, "w") if log else subprocess.DEVNULL
    return subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
