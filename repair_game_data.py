import argparse
import copy
import json
from pathlib import Path

SEASON_ORDER = {"S": 0, "F": 1, "W": 2}
PHASE_TYPE_ORDER = {"M": 0, "R": 1, "A": 2}


def contiguous_turn_blocks(samples):
    """Yield consecutive samples that belong to the same turn within a run."""
    start = 0
    while start < len(samples):
        turn = samples[start]["observation"]["turn"]
        end = start + 1
        while end < len(samples) and samples[end]["observation"]["turn"] == turn:
            end += 1
        yield samples[start:end]
        start = end


def phase_sort_key(turn):
    """Convert a Diplomacy phase like S1904R into a sortable key."""
    if not turn or len(turn) < 6:
        return None
    return (
        int(turn[1:5]),
        SEASON_ORDER.get(turn[0], 99),
        PHASE_TYPE_ORDER.get(turn[-1], 99),
    )


def contiguous_game_blocks(samples):
    """Yield consecutive samples that belong to the same game run."""
    if not samples:
        return

    start = 0
    previous_key = phase_sort_key(samples[0]["observation"].get("turn"))
    for idx in range(1, len(samples)):
        current_key = phase_sort_key(samples[idx]["observation"].get("turn"))
        if previous_key is not None and current_key is not None and current_key < previous_key:
            yield samples[start:idx]
            start = idx
        previous_key = current_key
    yield samples[start:]


def trim_public_chat(observation, window=None):
    """Drop future public-chat turns and optionally keep only the last N."""
    public_chat = observation.get("public_chat", [])
    turn = observation.get("turn")
    if not public_chat or not turn:
        return False, 0

    turn_to_index = {entry.get("turn"): idx for idx, entry in enumerate(public_chat)}
    if turn not in turn_to_index:
        return False, 0

    trimmed = copy.deepcopy(public_chat[: turn_to_index[turn] + 1])
    if window is not None:
        trimmed = trimmed[-window:]

    removed = len(public_chat) - len(trimmed)
    changed = trimmed != public_chat
    if changed:
        observation["public_chat"] = trimmed
    return changed, removed


def add_current_state_turn(observation):
    """Mirror the observation turn into current_state for schema consistency."""
    current_state = observation.setdefault("current_state", {})
    if current_state.get("turn") == observation.get("turn"):
        return False
    current_state["turn"] = observation.get("turn")
    return True


def backfill_target_communications(turn_block):
    """Restore the target player's metadata from sibling samples in the same turn."""
    known = {}
    for sample in turn_block:
        for power, info in sample["observation"].get("communications", {}).items():
            known.setdefault(power, copy.deepcopy(info))

    repaired = 0
    unresolved = []
    for sample in turn_block:
        observation = sample["observation"]
        target = observation.get("target_player")
        communications = observation.setdefault("communications", {})
        if target in communications:
            continue
        if target not in known:
            unresolved.append((observation.get("turn"), target))
            continue
        communications[target] = copy.deepcopy(known[target])
        repaired += 1
    return repaired, unresolved


def _metadata_score(turn_metadata):
    communications = turn_metadata.get("communications", {})
    communication_shift = turn_metadata.get("communication_shift", {})
    return len(communications) * 10 + len(communication_shift)


def backfill_history_communications(game_block):
    """Backfill per-turn history communication metadata within one game."""
    turn_metadata = {}
    for sample in game_block:
        observation = sample.get("observation", {})
        turn = observation.get("turn")
        if not turn:
            continue
        candidate = {
            "communications": copy.deepcopy(observation.get("communications", {})),
            "communication_shift": copy.deepcopy(observation.get("communication_shift", {})),
        }
        if turn not in turn_metadata or _metadata_score(candidate) > _metadata_score(turn_metadata[turn]):
            turn_metadata[turn] = candidate

    repaired_turns = 0
    repaired_samples = 0
    for sample in game_block:
        observation = sample.get("observation", {})
        history = observation.get("history", [])
        sample_changed = False
        for turn_state in history:
            turn = turn_state.get("turn")
            metadata = turn_metadata.get(turn)
            if not metadata:
                continue

            changed = False
            if metadata["communications"] and not turn_state.get("communications"):
                turn_state["communications"] = copy.deepcopy(metadata["communications"])
                changed = True
            if metadata["communication_shift"] and not turn_state.get("communication_shift"):
                turn_state["communication_shift"] = copy.deepcopy(metadata["communication_shift"])
                changed = True

            if changed:
                repaired_turns += 1
                sample_changed = True

        if sample_changed:
            repaired_samples += 1

    return repaired_samples, repaired_turns


def repair_game_data(data, public_chat_window=None):
    """Repair offline training data without regenerating games."""
    samples = data.get("training_data", [])
    stats = {
        "samples": len(samples),
        "public_chat_trimmed": 0,
        "public_chat_turns_removed": 0,
        "target_communications_backfilled": 0,
        "current_state_turn_added": 0,
        "history_samples_backfilled": 0,
        "history_turns_backfilled": 0,
        "unresolved_target_communications": [],
    }

    for sample in samples:
        observation = sample.get("observation", {})
        changed, removed = trim_public_chat(observation, window=public_chat_window)
        if changed:
            stats["public_chat_trimmed"] += 1
            stats["public_chat_turns_removed"] += removed
        if add_current_state_turn(observation):
            stats["current_state_turn_added"] += 1

    for block in contiguous_turn_blocks(samples):
        repaired, unresolved = backfill_target_communications(block)
        stats["target_communications_backfilled"] += repaired
        stats["unresolved_target_communications"].extend(unresolved)

    for game_block in contiguous_game_blocks(samples):
        repaired_samples, repaired_turns = backfill_history_communications(game_block)
        stats["history_samples_backfilled"] += repaired_samples
        stats["history_turns_backfilled"] += repaired_turns

    return data, stats


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}.repaired{input_path.suffix}")


def parse_args():
    parser = argparse.ArgumentParser(description="Repair generated Diplomacy training data.")
    parser.add_argument("input", nargs="?", default="game_data.json", help="Path to the source JSON file.")
    parser.add_argument(
        "--output",
        help="Path to write repaired JSON. Defaults to <input>.repaired.json unless --in-place is used.",
    )
    parser.add_argument(
        "--public-chat-window",
        type=int,
        default=None,
        help="Keep only the last N public-chat turns visible to each sample after trimming future turns.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a separate repaired file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print repair stats without writing a file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = input_path if args.in_place else Path(args.output) if args.output else default_output_path(input_path)

    with input_path.open() as f:
        data = json.load(f)

    repaired, stats = repair_game_data(data, public_chat_window=args.public_chat_window)

    print(f"Samples: {stats['samples']}")
    print(f"Trimmed public_chat in {stats['public_chat_trimmed']} samples")
    print(f"Removed {stats['public_chat_turns_removed']} future public-chat turns total")
    print(f"Backfilled target communications in {stats['target_communications_backfilled']} samples")
    print(f"Added current_state.turn in {stats['current_state_turn_added']} samples")
    print(f"Backfilled history communications in {stats['history_samples_backfilled']} samples")
    print(f"Filled {stats['history_turns_backfilled']} history turns with communication metadata")
    if stats["unresolved_target_communications"]:
        print(f"Unresolved target communications: {len(stats['unresolved_target_communications'])}")
        for turn, target in stats["unresolved_target_communications"][:10]:
            print(f"  {turn} {target}")

    if args.dry_run:
        print("Dry run only; no file written.")
        return

    with output_path.open("w") as f:
        json.dump(repaired, f, indent=2)

    print(f"Wrote repaired data to {output_path}")


if __name__ == "__main__":
    main()
