class_name DreamerController
extends AIController2D
## Dreamer's hands and eyes on party slot P1. It presses the same actions as a person
## or a PartyBot (p1_move_left, p1_move_right, p1_action_1), so it plays by the same rules.
##
## One episode = one race of P1: from "Go!" until P1 reaches the flag, is eliminated, or
## the race ends. RlSync only steps Python while `tick()` says so.

const SLOT := 1
const RAYS := 16            # walls and floors around P1
const RAY_LEN := 480.0
const DANGERS := 6          # nearest traps
const OTHERS := 3           # nearest other robots
const SCALE := 1000.0       # pixels -> about -1..1
const OBS_SIZE := 7 + RAYS + DANGERS * 6 + OTHERS * 2

## Rewards (same spirit as the party points: finish 10, first 5). Trying is worth more than
## waiting: in a 40 s race, waiting for the clock is about -3.4, getting 15 cells closer and then
## falling about -1, an early fall about -4, reaching the flag about +30.
const R_FINISH := 10.0
const R_FIRST := 5.0
const R_ELIMINATED := -5.0  # trap or pit
const R_TIMEOUT := -1.0     # still racing when the clock ran out
const R_PROGRESS := 0.3     # per cell closer to the flag than ever before in this race
const R_TIME := -0.001      # per physics frame (about -0.06 a second)

var ctl: Node               # the PartyController of the current party (set by RlTrainer)
var helper: HelperBot       # picks and places for P1, measures the distance to the flag
var in_episode := false
var episodes := 0

var _terminal := false
var _outcome := ""
var _cause := ""            # eliminated by: a trap's scene name, or "pit" (left the map)
var _trap_hit := ""         # the trap that touched P1 in this race, if any
var _best_dist := INF       # closest to the flag so far in this race (cells)
var _pressed := {}          # action name -> true


func _ready() -> void:
	super._ready()
	reset_after = 1 << 30  # episodes end with the race, not after a number of steps
	# Traps say whom they take out (Hazard.hit): listen to all of them, present and future.
	get_tree().node_added.connect(_watch_hazard)
	for n in get_tree().root.find_children("*", "Area2D", true, false):
		_watch_hazard(n)


# --- godot_rl_agents interface ----------------------------------------------------------

func get_action_space() -> Dictionary:
	return {
		"move": {"size": 3, "action_type": "discrete"},  # none, left, right
		"jump": {"size": 2, "action_type": "discrete"},  # released, held
	}


func set_action(action) -> void:
	if not in_episode or done:
		return
	var move := int(action["move"])
	_press("move_left", move == 1)
	_press("move_right", move == 2)
	_press("action_1", int(action["jump"]) == 1)


func get_obs() -> Dictionary:
	return {"obs": _features()}


func get_reward() -> float:
	return reward


func get_info() -> Dictionary:
	return {"terminal": _terminal, "outcome": _outcome, "cause": _cause, "episode": episodes}


func reset() -> void:
	super.reset()
	in_episode = false
	done = false
	reward = 0.0
	_release_all()


# --- Called by RlSync every physics frame -------------------------------------------------

## True while Python should be stepped: during P1's race, and until the end of it was sent.
func tick() -> bool:
	if needs_reset:
		reset()  # Python asked for a new episode: wait for the next race
	var p := _p1()
	if not in_episode:
		if not _racing(p):
			return false
		_start(p)
	if done:
		return true  # the end was not sent yet, or Python hasn't asked for a reset yet
	if not is_instance_valid(p) or p.get_meta("eliminated", false):
		if ctl.run_time >= _limit() - 0.1:
			_end(R_TIMEOUT, true, "timeout")  # the race eliminates whoever is still out there
		else:
			_cause = _trap_hit if _trap_hit else "pit"
			_end(R_ELIMINATED, true, "eliminated")
	elif p.get_meta("finished", false):
		var first: bool = not ctl.finished.is_empty() and ctl.finished[0] == SLOT
		_end(R_FINISH + (R_FIRST if first else 0.0), true, "first" if first else "finished")
	elif ctl.phase != ctl.Phase.RUN:
		_end(0.0, false, "race over")  # cut short (console skip, game mode): not P1's doing
	else:
		# Only new ground counts: no penalty for stepping back or falling, nothing to farm.
		var d := helper.distance(p.global_position) if helper else INF
		if is_finite(d) and d < _best_dist:
			if is_finite(_best_dist):
				reward += minf(R_PROGRESS * (_best_dist - d), 1.0)
			_best_dist = d
		reward += R_TIME
	return true


func _racing(p: Node) -> bool:
	return is_instance_valid(ctl) and ctl.phase == ctl.Phase.RUN and is_instance_valid(p) \
		and p.is_physics_processing() and not p.get_meta("finished", false) \
		and not p.get_meta("eliminated", false)


func _start(p: Node) -> void:
	in_episode = true
	done = false
	reward = 0.0
	_terminal = false
	_outcome = ""
	_cause = ""
	_trap_hit = ""
	_best_dist = helper.distance(p.global_position) if helper else INF


func _end(r: float, terminal: bool, outcome: String) -> void:
	reward += r
	done = true
	_terminal = terminal
	_outcome = outcome
	episodes += 1
	_release_all()


## The race's time limit in seconds (the party rule, else the arena's).
func _limit() -> float:
	return Party.rule("round_time") if Party.rule("round_time") > 0 else ctl.round_time


func _watch_hazard(n: Node) -> void:
	if n is Hazard and not n.hit.is_connected(_on_hazard_hit):
		n.hit.connect(_on_hazard_hit.bind(n))


func _on_hazard_hit(player: Node, hazard: Node) -> void:
	if in_episode and not done and player == _p1():
		_trap_hit = _trap_name(hazard)


## The trap's name: the placed item's id (saw, spikes...), else the hazard's own script for the
## ones made in code (crossbow bolts, planes, flamers, weather), else its parent's name.
func _trap_name(h: Node) -> String:
	var arena: Node = ctl.get_parent() if is_instance_valid(ctl) else null
	var n := h
	while n and n != arena:
		if n.has_meta("record"):
			return str(n.get_meta("record").get("id", "item"))
		n = n.get_parent()
	var path: String = h.get_script().resource_path if h.get_script() else ""
	if path and not path.ends_with("/hazard.gd"):
		return path.get_file().get_basename()  # "crossbow.gd::_Bolt" -> "crossbow"
	n = h.get_parent()
	while n and n != arena:
		if n.get_script() and n.get_script().resource_path:
			return n.get_script().resource_path.get_file().get_basename()  # flamer, arena_weather
		n = n.get_parent()
	return "trap"


func _p1() -> Node:
	if not is_instance_valid(ctl):
		return null
	var p = ctl.players.get(SLOT)
	return p if is_instance_valid(p) else null


# --- Keys ------------------------------------------------------------------------------

func _press(a: String, on: bool) -> void:
	var action := "p%d_%s" % [SLOT, a]
	if on and not _pressed.has(action):
		Input.action_press(action)
		_pressed[action] = true
	elif not on and _pressed.has(action):
		Input.action_release(action)
		_pressed.erase(action)


func _release_all() -> void:
	for action in _pressed:
		Input.action_release(action)
	_pressed.clear()


# --- Observation -----------------------------------------------------------------------

## OBS_SIZE floats, about -1..1:
##   7  P1: to the flag (x, y), velocity (x, y), on floor, distance-map cells / 100, race time left
##   16 rays: free distance around P1 (walls, floors, crates), 0 = touching, 1 = nothing
##   36 6 nearest traps: offset to the trap's rect center (x, y), its size (w, h), velocity (x, y)
##   6  3 nearest other robots: offset (x, y)
func _features() -> Array:
	var out: Array = []
	out.resize(OBS_SIZE)
	out.fill(0.0)
	var p := _p1() as CharacterBody2D
	if p == null:
		return out
	var pos: Vector2 = p.global_position
	var i := 0
	var to_goal: Vector2 = ctl._goal.global_position - pos
	var d := helper.distance(pos) if helper else INF
	var limit := _limit()
	for v in [to_goal.x / SCALE, to_goal.y / SCALE, p.velocity.x / SCALE, p.velocity.y / SCALE,
			1.0 if p.is_on_floor() else 0.0, (d / 100.0) if is_finite(d) else 1.0,
			clampf(1.0 - ctl.run_time / limit, 0.0, 1.0)]:
		out[i] = v
		i += 1
	# Rays.
	var space: PhysicsDirectSpaceState2D = p.get_world_2d().direct_space_state
	for k in RAYS:
		var dir := Vector2.RIGHT.rotated(TAU * k / RAYS)
		var q := PhysicsRayQueryParameters2D.create(pos, pos + dir * RAY_LEN, Layers.WORLD | Layers.CRATES)
		q.exclude = [p.get_rid()]
		var hit: Dictionary = space.intersect_ray(q)
		out[i] = (pos.distance_to(hit.position) / RAY_LEN) if hit else 1.0
		i += 1
	# Traps, nearest first.
	var traps: Array = helper.dangers(p) if helper else []
	traps.sort_custom(func(a, b): return _rect_gap(a[0], pos) < _rect_gap(b[0], pos))
	for k in DANGERS:
		if k < traps.size():
			var r: Rect2 = traps[k][0]
			var v: Vector2 = traps[k][1]
			for x in [(r.get_center().x - pos.x) / SCALE, (r.get_center().y - pos.y) / SCALE,
					r.size.x / SCALE, r.size.y / SCALE, v.x / SCALE, v.y / SCALE]:
				out[i] = x
				i += 1
		else:
			for x in [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]:  # "far away"
				out[i] = x
				i += 1
	# Other robots, nearest first.
	var others: Array = []
	for s in ctl.players:
		var o = ctl.players[s]
		if s != SLOT and is_instance_valid(o) and not o.get_meta("eliminated", false):
			others.append(o.global_position - pos)
	others.sort_custom(func(a, b): return a.length() < b.length())
	for k in OTHERS:
		var off: Vector2 = others[k] if k < others.size() else Vector2(SCALE, SCALE)
		out[i] = off.x / SCALE
		out[i + 1] = off.y / SCALE
		i += 2
	return out


static func _rect_gap(r: Rect2, pos: Vector2) -> float:
	return (pos.clamp(r.position, r.end) - pos).length()
