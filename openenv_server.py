import json
import os
import re

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

from prompts import JUDGE_PROMPT

app = FastAPI()
judge_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
JUDGE_MODEL = "claude-haiku-4-5-20251001"

with open("game_data.json", "r") as f:
    ALL_SAMPLES = json.load(f)["training_data"]

idx = 0


async def _judge(power, true_strategy, predicted_strategy):
    prompt = JUDGE_PROMPT.format(
        power=power,
        true_strategy=true_strategy,
        predicted_strategy=predicted_strategy,
    )
    response = await judge_client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=40,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
<<<<<<< HEAD
    if "SCORE: 1" in text:
        return 1
=======
    score_match = re.search(r"SCORE:\s*([01])\b", text, re.IGNORECASE)
    if score_match:
        return int(score_match.group(1))

    # Fall back to a bare 0/1 if the model ignores formatting.
    if text in {"0", "1"}:
        return int(text)

    # Fail closed if the model returns an unexpected format.
>>>>>>> cb6df70 (Unify demo UI and judge scoring flow)
    return 0


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset():
    global idx
    idx = 0
    return {"observation": ALL_SAMPLES[idx]["observation"]}


class StepRequest(BaseModel):
    target_player: str
    prediction: str


@app.post("/step")
async def step(req: StepRequest):
    global idx
    target = req.target_player.upper()
    true_strategy = ALL_SAMPLES[idx]["true_strategy"]

    reward = await _judge(target, true_strategy, req.prediction)

    idx += 1
    done = idx >= len(ALL_SAMPLES)
    observation = ALL_SAMPLES[idx]["observation"] if not done else {}

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
