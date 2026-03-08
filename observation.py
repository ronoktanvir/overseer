import copy


POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def _compute_communication_shift(comm_tracker):
    """Compare curr vs prev message counts to detect diplomatic shifts."""
    curr = comm_tracker.get("curr", {})
    prev = comm_tracker.get("prev", {})
    shift = {}

    for power in POWERS:
        curr_count = curr.get(power, {}).get("message_count", 0)
        prev_count = prev.get(power, {}).get("message_count", 0)
        delta = curr_count - prev_count
        if delta != 0:
            shift[power] = delta

    return shift


def _snapshot_communications(comm_tracker):
    """Serialize current-turn communication metadata for storage in history."""
    curr_comm = (comm_tracker or {}).get("curr", {})
    communication = {}
    for power in POWERS:
        info = curr_comm.get(power, {})
        communication[power] = {
            "messaged": list(info.get("messaged", [])),
            "ignored": list(info.get("ignored", [])),
            "message_count": info.get("message_count", 0),
        }
    return communication


def _snapshot_public_chat(public_chat, last_n=5):
    """Freeze public-chat history so later turns cannot leak backward."""
    recent_turns = public_chat[-last_n:] if last_n else public_chat
    return [
        {
            "turn": turn.get("turn"),
            "messages": [[speaker, message] for speaker, message in turn.get("messages", [])],
        }
        for turn in recent_turns
    ]


def build_current_state(game, phase=None, submitted_orders=None, comm_tracker=None):
    """Extract the current board state from a diplomacy Game object."""
    units = {}
    supply_centers = {}
    orders = {}
    submitted_orders = submitted_orders or {}

    for power in POWERS:
        p = game.powers[power]
        units[power] = list(p.units)
        supply_centers[power] = list(p.centers)
        order_snapshot = submitted_orders[power] if power in submitted_orders else game.get_orders(power)
        orders[power] = list(order_snapshot)

    return {
        "turn": phase or game.get_current_phase(),
        "units": units,
        "supply_centers": supply_centers,
        "orders": orders,
        "conflicts": {},
        "communications": _snapshot_communications(comm_tracker),
        "communication_shift": _compute_communication_shift(comm_tracker or {}),
    }


def build_observation(target_player, current_state, history, comm_tracker, public_chat):
    """Assemble the full observation dict for the overseer."""
    communication = current_state.get("communications") or _snapshot_communications(comm_tracker)
    communication_shift = current_state.get("communication_shift") or _compute_communication_shift(comm_tracker)

    return {
        "target_player": target_player,
        "turn": current_state["turn"],
        "current_state": {
            "turn": current_state["turn"],
            "units": current_state["units"],
            "supply_centers": current_state["supply_centers"],
            "orders": current_state["orders"],
            "conflicts": current_state["conflicts"],
        },
        "history": copy.deepcopy(history[-5:]),
        "communications": communication,
        "communication_shift": communication_shift,
        "public_chat": _snapshot_public_chat(public_chat),
    }
