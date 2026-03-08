import asyncio
import json
import os
import re

import anthropic

from diplomacy import Game
from prompts import PLAYER_PROMPT
from observation import POWERS, build_current_state, build_observation

MODEL = "claude-haiku-4-5-20251001"
client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_connection_semaphore = asyncio.Semaphore(7)


def _parse_section(text, header):
    """Extract the content after a HEADER: line up to the next known header or end."""
    headers = ["ORDERS:", "STRATEGY:", "PRIVATE MESSAGES:", "PUBLIC MESSAGE:"]
    pattern = re.escape(header) + r"\s*\n(.*?)(?=" + "|".join(re.escape(h) for h in headers if h != header) + r"|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _parse_private_messages(raw):
    """Parse 'TO POWER: message' lines into a dict."""
    messages = {}
    if not raw or raw.lower() == "none":
        return messages
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"TO\s+(\w+):\s*(.*)", line, re.IGNORECASE)
        if m:
            messages[m.group(1).upper()] = m.group(2).strip()
    return messages


def _format_board_state(game):
    """Readable string of all units on the board."""
    lines = []
    for power in POWERS:
        units = game.powers[power].units
        if units:
            lines.append(f"{power}: {', '.join(units)}")
    return "\n".join(lines)


def _format_all_centers(game):
    """Readable string of supply center ownership."""
    lines = []
    for power in POWERS:
        centers = game.powers[power].centers
        if centers:
            lines.append(f"{power}: {', '.join(sorted(centers))}")
    return "\n".join(lines)


def _format_possible_orders(game, power):
    """Filter possible orders to only those relevant to this power's units."""
    all_possible = game.get_all_possible_orders()
    unit_locs = {u.split()[-1] for u in game.powers[power].units}
    lines = []
    for loc in sorted(unit_locs):
        if loc in all_possible and all_possible[loc]:
            lines.append(f"{loc}: {', '.join(all_possible[loc])}")
    return "\n".join(lines) if lines else "No orders available."


def _validate_orders(game, power, orders):
    """Filter orders to only those the game engine accepts. Invalid ones get HOLD."""
    all_possible = game.get_all_possible_orders()
    valid_set = set()
    for loc_orders in all_possible.values():
        valid_set.update(loc_orders)

    valid = []
    unit_locs_covered = set()
    for order in orders:
        if order in valid_set:
            valid.append(order)
            parts = order.split()
            if len(parts) >= 2:
                unit_locs_covered.add(parts[1])

    for unit in game.powers[power].units:
        loc = unit.split()[-1]
        if loc not in unit_locs_covered:
            valid.append(f"{unit} H")

    return valid


def _format_messages_for_power(power, private_messages_this_turn, public_messages_this_turn):
    """Build the messages string a player sees this phase."""
    lines = []
    for sender, msg_dict in private_messages_this_turn.items():
        if power in msg_dict:
            lines.append(f"FROM {sender}: {msg_dict[power]}")
    for sender, msg in public_messages_this_turn:
        lines.append(f"PUBLIC ({sender}): {msg}")
    return "\n".join(lines) if lines else "No messages this phase."


def _format_history(history, last_n=3):
    """Summarize the last N turns of history for the prompt."""
    recent = history[-last_n:] if history else []
    if not recent:
        return "No prior history."
    lines = []
    for state in recent:
        lines.append(f"--- {state['turn']} ---")
        for p in POWERS:
            orders = state["orders"].get(p, [])
            if orders:
                lines.append(f"  {p}: {', '.join(orders)}")
    return "\n".join(lines)


async def call_player_agent(power, game, private_messages_this_turn, public_messages_this_turn, history):
    """Call Claude Haiku for a single power and parse the structured response."""
    prompt = PLAYER_PROMPT.format(
        power=power,
        phase=game.get_current_phase(),
        units=", ".join(game.powers[power].units),
        centers=", ".join(sorted(game.powers[power].centers)),
        all_centers=_format_all_centers(game),
        board_state=_format_board_state(game),
        history=_format_history(history),
        messages=_format_messages_for_power(power, private_messages_this_turn, public_messages_this_turn),
        possible_orders=_format_possible_orders(game, power),
    )

    async with _connection_semaphore:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    text = response.content[0].text

    orders_raw = _parse_section(text, "ORDERS:")
    orders = [line.strip() for line in orders_raw.splitlines() if line.strip()]

    strategy = _parse_section(text, "STRATEGY:")
    if not strategy:
        print(f"  Warning: {power} returned empty strategy, using fallback")
        strategy = "No strategy provided"
    private_msgs = _parse_private_messages(_parse_section(text, "PRIVATE MESSAGES:"))
    public_msg = _parse_section(text, "PUBLIC MESSAGE:")
    if public_msg.lower() == "none":
        public_msg = ""

    return {
        "power": power,
        "orders": orders,
        "strategy": strategy,
        "private_messages": private_msgs,
        "public_message": public_msg,
    }


def _init_comm_tracker():
    return {"curr": {p: {"messaged": set(), "ignored": set(), "message_count": 0} for p in POWERS}, "prev": {}}


def _update_comm_tracker(comm_tracker, results):
    """Rotate curr -> prev, then rebuild curr from this turn's messages."""
    comm_tracker["prev"] = {
        p: {
            "messaged": set(comm_tracker["curr"][p]["messaged"]),
            "ignored": set(comm_tracker["curr"][p]["ignored"]),
            "message_count": comm_tracker["curr"][p]["message_count"],
        }
        for p in POWERS
    }
    for p in POWERS:
        comm_tracker["curr"][p] = {"messaged": set(), "ignored": set(), "message_count": 0}

    all_powers_messaged_to = {p: set() for p in POWERS}
    for r in results:
        sender = r["power"]
        recipients = set(r["private_messages"].keys())
        all_powers_messaged_to[sender] = recipients
        comm_tracker["curr"][sender]["message_count"] = len(recipients)
        comm_tracker["curr"][sender]["messaged"] = recipients

    for p in POWERS:
        others = set(POWERS) - {p}
        messaged_to = all_powers_messaged_to[p]
        comm_tracker["curr"][p]["ignored"] = others - messaged_to


def _serializable_comm_tracker(comm_tracker):
    """Convert sets to lists for JSON serialization."""
    out = {}
    for key in ("curr", "prev"):
        out[key] = {}
        for p, data in comm_tracker.get(key, {}).items():
            out[key][p] = {
                "messaged": sorted(data["messaged"]) if isinstance(data["messaged"], set) else data["messaged"],
                "ignored": sorted(data["ignored"]) if isinstance(data["ignored"], set) else data["ignored"],
                "message_count": data["message_count"],
            }
    return out


async def run_game(max_turns=20):
    game = Game()
    history = []
    training_data = []
    public_chat_log = []
    comm_tracker = _init_comm_tracker()

    private_messages_this_turn = {}
    public_messages_this_turn = []

    for turn in range(max_turns):
        phase = game.get_current_phase()
        print(f"=== Turn {turn + 1}: {phase} ===")

        if game.is_game_done:
            print("Game over.")
            break

        results = await asyncio.gather(*(
            call_player_agent(p, game, private_messages_this_turn, public_messages_this_turn, history)
            for p in POWERS
            if game.powers[p].units
        ))

        private_messages_this_turn = {}
        public_messages_this_turn = []
        strategies = {}
        submitted_orders = {power: [] for power in POWERS}

        for r in results:
            power = r["power"]
            validated = _validate_orders(game, power, r["orders"])
            submitted_orders[power] = list(validated)
            game.set_orders(power, validated)
            strategies[power] = r["strategy"]
            if r["private_messages"]:
                private_messages_this_turn[power] = r["private_messages"]
            if r["public_message"]:
                public_messages_this_turn.append((power, r["public_message"]))

        _update_comm_tracker(comm_tracker, results)

        public_chat_log.append({
            "turn": phase,
            "messages": [(s, m) for s, m in public_messages_this_turn],
        })

        current_state = build_current_state(game, phase=phase, submitted_orders=submitted_orders)
        history.append(current_state)

        game.process()

        if turn >= 4:
            for power in POWERS:
                if not game.powers[power].units:
                    continue
                obs = build_observation(
                    power, current_state, history,
                    _serializable_comm_tracker(comm_tracker),
                    public_chat_log,
                )
                training_data.append({
                    "observation": obs,
                    "true_strategy": strategies.get(power, ""),
                })

    existing = {"training_data": [], "public_chat_log": [], "history": []}
    if os.path.exists("game_data.json"):
        with open("game_data.json", "r") as f:
            existing = json.load(f)

    existing["training_data"].extend(training_data)
    existing["public_chat_log"].extend(public_chat_log)
    existing["history"].extend(history)

    with open("game_data.json", "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"Saved {len(training_data)} new samples ({len(existing['training_data'])} total) to game_data.json")
    return existing


if __name__ == "__main__":
    asyncio.run(run_game())
