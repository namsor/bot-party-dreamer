extends SceneTree
## Probe: run alongside the bot-party command line and report who is in the party.
##   godot --headless --path C:/git/bot-party --script C:/git/bot-party-dreamer/tools/probe_party.gd -- --party --players=1 --bots=3

var _frames := 0

func _physics_process(_d: float) -> bool:
	_frames += 1
	if _frames % 300 != 0:
		return false
	var party := root.get_node("Party")
	var ctl := get_first_node_in_group("party_controller")
	var players := get_nodes_in_group("players")
	print("t=%ds joined=%s bots=%s phase=%s players=%s" % [_frames / 60, root.get_node("GameState").joined,
		party.bots(), ctl.phase if ctl else "-", players.map(func(p): return p.name)])
	if _frames >= 1800:
		quit()
	return false
