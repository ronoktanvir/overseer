import json
import os
import pathlib
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = pathlib.Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = BASE_DIR / "overseer" / "frontend"
POWER_ORDER = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="static",
    )

training_data = []
turn_groups = []
current_turn_index = 0
last_reward = None
reward_history = []
initialization_error = None
data_path = None


class AdvanceRequest(BaseModel):
    prediction: str = ""


def _candidate_data_paths():
    env_path = os.environ.get("OVERSEER_DATA_PATH")
    if env_path:
        yield pathlib.Path(env_path).expanduser()
    yield BASE_DIR / "game_data.json"
    yield BASE_DIR / "overseer" / "game_data.json"
    yield pathlib.Path.home() / "overseer" / "game_data.json"


def _load_training_data() -> None:
    global training_data, data_path

    for candidate in _candidate_data_paths():
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded_training_data = payload.get("training_data") or []
        if loaded_training_data:
            training_data = loaded_training_data
            data_path = str(candidate)
            return

    raise HTTPException(
        status_code=500,
        detail="Unable to locate game_data.json. Set OVERSEER_DATA_PATH or place the file in an expected location.",
    )


def _count_entries(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value)
    return 0


def _get_power_map(name: str, observation: dict) -> dict:
    current_state = observation.get("current_state") or {}
    return current_state.get(name) or {}


def _shorten_strategy(text: str, word_limit: int = 20) -> str:
    if not text:
        return "No strategy provided."
    sentence = text.split(".")[0].strip()
    words = sentence.split()
    if len(words) <= word_limit:
        return sentence if sentence.endswith(".") else sentence + "."
    return " ".join(words[:word_limit]).rstrip(",") + "..."


def _order_style_summary(orders: list[str]) -> str:
    if not orders:
        return "Defensive consolidation around core territory."
    support_count = sum(" S " in order for order in orders)
    move_count = sum(" - " in order for order in orders)
    hold_count = sum(" H" in order for order in orders)

    if support_count >= max(1, len(orders) // 2):
        return "Support-heavy coordination to lock down borders and back allied moves."
    if move_count >= max(1, len(orders) // 2):
        return "Expansion-focused positioning toward contested territory and leverage points."
    if hold_count >= max(1, len(orders) // 2):
        return "Static defensive posture while preserving flexibility for the next phase."
    return "Measured repositioning to preserve tempo without overcommitting."


def _wrong_prediction(power: str, orders: list[str], turn_index: int, power_index: int) -> str:
    templates = [
        f"{power.title()} is likely prioritizing a cautious defensive hold around its home centers.",
        f"{power.title()} appears to be probing for expansion while avoiding a firm alliance commitment.",
        _order_style_summary(orders),
        f"{power.title()} is probably signaling restraint while quietly setting up a later offensive swing.",
    ]
    return templates[(turn_index + power_index) % len(templates)]


def _mock_turn_card(sample: dict, previous_sample: dict | None, turn_index: int, power_index: int) -> dict:
    observation = sample.get("observation") or {}
    power = (observation.get("target_player") or "UNKNOWN").upper()
    true_strategy = sample.get("true_strategy") or "No strategy provided."
    units = _get_power_map("units", observation)
    supply_centers = _get_power_map("supply_centers", observation)
    orders_map = _get_power_map("orders", observation)
    orders = orders_map.get(power) or []
    previous_observation = (previous_sample or {}).get("observation") or {}
    previous_units = _get_power_map("units", previous_observation)
    previous_supply_centers = _get_power_map("supply_centers", previous_observation)
    unit_count = _count_entries(units.get(power))
    supply_center_count = _count_entries(supply_centers.get(power))
    previous_unit_count = _count_entries(previous_units.get(power))
    previous_supply_center_count = _count_entries(previous_supply_centers.get(power))

    judged_correct = ((turn_index * 3 + power_index) % 5) not in (0,)
    predicted_strategy = (
        _shorten_strategy(true_strategy, 18)
        if judged_correct
        else _wrong_prediction(power, orders, turn_index, power_index)
    )

    return {
        "power": power,
        "unit_count": unit_count,
        "previous_unit_count": previous_unit_count,
        "unit_delta": unit_count - previous_unit_count,
        "supply_center_count": supply_center_count,
        "previous_supply_center_count": previous_supply_center_count,
        "supply_center_delta": supply_center_count - previous_supply_center_count,
        "orders": orders,
        "actual_strategy": true_strategy,
        "actual_strategy_short": _shorten_strategy(true_strategy, 22),
        "predicted_strategy": predicted_strategy,
        "predicted_strategy_short": _shorten_strategy(predicted_strategy, 18),
        "judge_correct": judged_correct,
    }


def _build_turn_groups() -> None:
    global turn_groups

    turn_groups = []
    current_turn = None
    current_samples = []

    for sample in training_data:
        turn = (sample.get("observation") or {}).get("turn") or "UNKNOWN"
        if current_turn is None or turn == current_turn:
            current_turn = turn
            current_samples.append(sample)
            continue

        turn_groups.append({"turn": current_turn, "samples": current_samples})
        current_turn = turn
        current_samples = [sample]

    if current_samples:
        turn_groups.append({"turn": current_turn, "samples": current_samples})


def _current_turn_payload() -> dict:
    if not turn_groups:
        raise HTTPException(status_code=500, detail="Turn groups are empty.")

    turn_group = turn_groups[current_turn_index]
    previous_turn_group = turn_groups[current_turn_index - 1] if current_turn_index > 0 else None
    previous_samples = {}
    if previous_turn_group:
        previous_samples = {
            ((sample.get("observation") or {}).get("target_player") or "UNKNOWN").upper(): sample
            for sample in previous_turn_group["samples"]
        }
    cards = [
        _mock_turn_card(
            sample,
            previous_samples.get(((sample.get("observation") or {}).get("target_player") or "UNKNOWN").upper()),
            current_turn_index,
            index
        )
        for index, sample in enumerate(sorted(
            turn_group["samples"],
            key=lambda sample: POWER_ORDER.index(((sample.get("observation") or {}).get("target_player") or "UNKNOWN").upper())
            if ((sample.get("observation") or {}).get("target_player") or "UNKNOWN").upper() in POWER_ORDER else 999
        ))
    ]
    correct_count = sum(1 for card in cards if card["judge_correct"])
    turn_reward = 1 if correct_count * 2 >= len(cards) else 0

    return {
        "turn_label": turn_group["turn"],
        "turn_number": current_turn_index + 1,
        "turn_count": len(turn_groups),
        "countries": cards,
        "correct_count": correct_count,
        "country_count": len(cards),
        "turn_reward": turn_reward,
    }


def _overseer_state() -> dict:
    turn_payload = _current_turn_payload()
    countries = turn_payload["countries"]

    return {
        "turn": turn_payload["turn_label"],
        "turn_number": turn_payload["turn_number"],
        "turn_count": turn_payload["turn_count"],
        "countries": countries,
        "correct_count": turn_payload["correct_count"],
        "country_count": turn_payload["country_count"],
        "reward": last_reward,
        "reward_history": list(reward_history),
        "data_path": data_path,
        "judge_ready": False,
        "initialization_error": initialization_error,
    }


def _set_current_turn(index: int) -> None:
    global current_turn_index, last_reward

    if not turn_groups:
        raise HTTPException(status_code=500, detail="Turn groups are empty.")

    current_turn_index = index % len(turn_groups)
    last_reward = _current_turn_payload()["turn_reward"]


def _reset_session() -> None:
    global reward_history, initialization_error
    _load_training_data()
    _build_turn_groups()
    _set_current_turn(0)
    reward_history = [last_reward]
    initialization_error = None


def _ensure_session_initialized() -> None:
    global initialization_error
    if turn_groups:
        return
    try:
        _reset_session()
    except HTTPException as exc:
        initialization_error = exc.detail
        raise


@app.on_event("startup")
def startup_reset():
    global initialization_error
    try:
        _reset_session()
    except HTTPException as exc:
        initialization_error = exc.detail


@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Overseer frontend not found. Expected index.html in frontend/ or overseer/frontend/."}


@app.post("/reset")
def reset_demo():
    _reset_session()
    return {"overseer_state": _overseer_state()}


@app.get("/overseer_state")
def overseer_state():
    _ensure_session_initialized()
    return _overseer_state()


@app.post("/advance")
def advance(request: AdvanceRequest):
    _ensure_session_initialized()

    wrapped = current_turn_index >= len(turn_groups) - 1
    _set_current_turn(0 if wrapped else current_turn_index + 1)
    reward_history.append(last_reward)
    reward_history[:] = reward_history[-10:]

    return {
        "done": wrapped,
        "wrapped": wrapped,
        "prediction": request.prediction.strip(),
        "overseer_state": _overseer_state(),
    }


@app.post("/previous")
def previous():
    _ensure_session_initialized()

    wrapped = current_turn_index <= 0
    _set_current_turn(len(turn_groups) - 1 if wrapped else current_turn_index - 1)

    return {
        "wrapped": wrapped,
        "overseer_state": _overseer_state(),
    }
