POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def build_current_state(game, phase=None, submitted_orders=None):
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
    }


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


def build_observation(target_player, current_state, history, comm_tracker, public_chat):
    """Assemble the full observation dict for the overseer."""
    curr_comm = comm_tracker.get("curr", {})

    communication = {}
    for power in POWERS:
        if power == target_player:
            continue
        info = curr_comm.get(power, {})
        communication[power] = {
            "messaged": info.get("messaged", False),
            "ignored": info.get("ignored", False),
            "message_count": info.get("message_count", 0),
        }

    return {
        "target_player": target_player,
        "turn": current_state["turn"],
        "current_state": {
            "units": current_state["units"],
            "supply_centers": current_state["supply_centers"],
            "orders": current_state["orders"],
            "conflicts": current_state["conflicts"],
        },
        "history": history[-5:],
        "communications": communication,
        "communication_shift": _compute_communication_shift(comm_tracker),
        "public_chat": public_chat,
    }
