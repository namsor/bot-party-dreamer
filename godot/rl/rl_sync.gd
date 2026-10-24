class_name RlSync
extends Sync
## godot_rl_agents' Sync, but it only talks to Python while Dreamer races: picking,
## placing, the countdown and the scores run on their own between two episodes.

var agent: DreamerController


func _physics_process(delta: float) -> void:
	if agent and not agent.tick():
		return
	super._physics_process(delta)
