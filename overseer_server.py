from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pathlib
import random

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

reward_history = []

POWERS = ["FRANCE", "GERMANY", "RUSSIA", "AUSTRIA", "TURKEY", "ITALY", "ENGLAND"]
TURN_SEQUENCE = [
    "S1901M", "F1901M",
    "S1901R", "F1901R",
    "S1902M", "F1902M",
    "S1902R", "F1902R",
]
current_turn_index = 5

BASE_DIR = pathlib.Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "overseer" / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="static",
    )


@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Overseer frontend not found. Expected index.html in overseer/frontend/."}

@app.get("/overseer_state")
def overseer():
    r = random.randint(0, 1)
    reward_history.append(r)
    if len(reward_history) > 10:
        reward_history.pop(0)
    turn = TURN_SEQUENCE[current_turn_index % len(TURN_SEQUENCE)]
    prev_turn = TURN_SEQUENCE[(current_turn_index - 1) % len(TURN_SEQUENCE)]

    predictions = {
        "FRANCE": "Allied with ENGLAND",
        "GERMANY": "Preparing eastern expansion",
        "RUSSIA": "Moving toward BLACK SEA",
        "AUSTRIA": "Defensive posture",
        "TURKEY": "Opportunistic — watching RUSSIA",
        "ITALY": "Neutral",
        "ENGLAND": "Naval dominance strategy",
    }

    # Simple placeholders for now; later these should be driven by real
    # game state (supply centers, threat evaluation, deception checks).
    supply_centers = {
        "FRANCE": 4,
        "GERMANY": 5,
        "RUSSIA": 6,
        "AUSTRIA": 4,
        "TURKEY": 5,
        "ITALY": 3,
        "ENGLAND": 5,
    }
    threat_levels = {
        "FRANCE": "MED",
        "GERMANY": "HIGH",
        "RUSSIA": "HIGH",
        "AUSTRIA": "MED",
        "TURKEY": "MED",
        "ITALY": "LOW",
        "ENGLAND": "LOW",
    }
    deception_flags = {
        power: bool(random.randint(0, 4) == 0)
        for power in POWERS
    }

    last_turn_predictions = predictions
    last_turn_actual = {
        "FRANCE": "Lean toward ENGLAND, probes GERMANY",
        "GERMANY": "Builds army, eyes BELGIUM",
        "RUSSIA": "Aggressive push into BLACK SEA",
        "AUSTRIA": "Holds core, negotiates with ITALY",
        "TURKEY": "Plays patiently, builds fleets",
        "ITALY": "Balances between FRANCE and AUSTRIA",
        "ENGLAND": "Secures North Sea, hedges vs FRANCE",
    }
    last_turn_correct = {
        power: last_turn_predictions[power].split()[0] == last_turn_actual[power].split()[0]
        for power in POWERS
    }

    return {
        "turn": turn,
        "predictions": predictions,
        "reward": r,
        "reward_history": list(reward_history),
        "supply_centers": supply_centers,
        "threat_levels": threat_levels,
        "deception_flags": deception_flags,
        "last_turn": {
            "turn": prev_turn,
            "predictions": last_turn_predictions,
            "actual_strategies": last_turn_actual,
            "correct": last_turn_correct,
        },
    }
