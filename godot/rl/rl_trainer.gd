class_name RlTrainer
extends Node
## Started by the command line option --rl: Dreamer (Python, godot_rl_agents) plays P1
## of a party against the bots. Lives under /root, so it outlasts scene changes:
##   - Sync (RlSync): talks to Python while P1 races;
##   - Dreamer (DreamerController): P1's keys, observations and rewards;
##   - a HelperBot in each party: picks and places for P1;
##   - when a party is over, starts a new one.

const RESTART_AFTER := 3.0  # seconds without a party before starting a new one

var sync: RlSync
var agent: DreamerController
var _no_party := 0.0


func _init() -> void:
	name = "RlTrainer"
	process_mode = Node.PROCESS_MODE_ALWAYS  # Sync pauses the tree while it waits for Python
	agent = DreamerController.new()
	agent.name = "Dreamer"
	add_child(agent)
	sync = RlSync.new()
	sync.name = "Sync"
	sync.agent = agent
	add_child(sync)


func _process(delta: float) -> void:
	var ctl := get_tree().get_first_node_in_group("party_controller")
	if ctl != agent.ctl:
		agent.ctl = ctl
		agent.helper = null
		if ctl:
			var helper := HelperBot.new(DreamerController.SLOT, PartyBot.Skill.HARD)
			helper.ctl = ctl
			ctl.add_child(helper)
			agent.helper = helper
	if ctl:
		_no_party = 0.0
		return
	_no_party += delta
	if _no_party > RESTART_AFTER:
		_no_party = 0.0
		print("RL: new party")
		Party.new_game()
