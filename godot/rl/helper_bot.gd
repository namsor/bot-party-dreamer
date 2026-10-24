class_name HelperBot
extends PartyBot
## Picks and places for Dreamer's slot, but never races: in the race, Dreamer drives.
## Dreamer also borrows its distance map (`_dist`) and trap finder (`_find_dangers`).


func _physics_process(_delta: float) -> void:
	pass  # PartyBot races here; the race is Dreamer's


## Distance to the flag in cells (INF when the map isn't made yet or `pos` is off it).
func distance(pos: Vector2) -> float:
	if not is_instance_valid(ctl):
		return INF
	if _field_key == "":
		_make_field()
	return _dist(pos)


## Traps near the race: [Rect2 (grown by the body size), velocity, ...].
func dangers(p: Node) -> Array:
	return _find_dangers(p) if is_instance_valid(ctl) else []
