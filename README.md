---
title: Overseer OpenEnv
emoji: "♟️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Diplomacy Overseer

OpenEnv environment for training an overseer model to infer hidden Diplomacy player strategies from public state, order history, communication metadata, and public messages.

Tracks:
- Statement 1: Multi-Agent Interactions
- Fleet AI bonus: Scalable Oversight

## What is in this repo

- [server/app.py](/Users/ronoktanvir/Documents/openenv-hacks/server/app.py): real OpenEnv 0.2.1 environment server
- [server/overseer_environment.py](/Users/ronoktanvir/Documents/openenv-hacks/server/overseer_environment.py): offline environment over `game_data.json`
- [game_data.json](/Users/ronoktanvir/Documents/openenv-hacks/game_data.json): repaired training data with `game_id` and `game_step_index`
- [judge.py](/Users/ronoktanvir/Documents/openenv-hacks/judge.py): binary strategy similarity judge
- [overseer_server.py](/Users/ronoktanvir/Documents/openenv-hacks/overseer_server.py): demo dashboard server
- [training/minimal_trl_sft.py](/Users/ronoktanvir/Documents/openenv-hacks/training/minimal_trl_sft.py): minimal HF TRL training script
- [training/minimal_trl_grpo.py](/Users/ronoktanvir/Documents/openenv-hacks/training/minimal_trl_grpo.py): minimal HF TRL GRPO script with binary judge reward
- [colab/minimal_trl_overseer.ipynb](/Users/ronoktanvir/Documents/openenv-hacks/colab/minimal_trl_overseer.ipynb): Colab notebook wrapper for training

## Real OpenEnv API

Run:

```bash
./.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8001
```

Useful endpoints:

- `GET /health`
- `GET /schema`
- `GET /metadata`
- `GET /state`
- `POST /reset`
- `POST /step`

Example:

```bash
curl -X POST http://localhost:8001/reset -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"prediction":"Austria is trying to stabilize its eastern position while coordinating with Germany and watching Turkey.","target_player":"AUSTRIA"}}'
```

Note:
- `/step` requires `ANTHROPIC_API_KEY` because reward scoring uses the binary judge.

## Demo Dashboard

Run:

```bash
./.venv/bin/uvicorn overseer_server:app --host 0.0.0.0 --port 8002
```

Open:

- `http://localhost:8002`

This is only for visualization. It is not the actual OpenEnv server.

## Training

Minimal TRL script:

```bash
python training/minimal_trl_sft.py \
  --data-path game_data.json \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/minimal_trl_sft \
  --num-train-epochs 1 \
  --max-eval-samples 8 \
  --judge-eval
```

What it does:

- splits train/eval by `game_id`
- fine-tunes with HF TRL `SFTTrainer`
- logs before/after evaluation
- uses the real binary judge if `ANTHROPIC_API_KEY` is set
- otherwise falls back to a cheap token-overlap proxy

Minimal GRPO script:

```bash
python -m training.minimal_trl_grpo \
  --data-path game_data.json \
  --output-dir outputs/minimal_trl_grpo \
  --max-steps 20 \
  --num-generations 2
```

What it does:

- samples completions from the overseer model
- scores them with the binary Claude judge
- updates the model with HF TRL `GRPOTrainer`
- saves reward logs to `outputs/minimal_trl_grpo/metrics.json`

Colab notebook:

- [colab/minimal_trl_overseer.ipynb](/Users/ronoktanvir/Documents/openenv-hacks/colab/minimal_trl_overseer.ipynb)

## Data Notes

Each observation includes:

- `game_id`
- `game_step_index`
- `turn`
- `target_player`
- `current_state`
- `history` (last 5 turns only)
- `communications`
- `communication_shift`
- `public_chat` (last 5 turns only)

This makes it easy to accumulate all steps belonging to one game during training or evaluation.

## Hugging Face Spaces

This repo is packaged for OpenEnv/HF Spaces deployment with:

- [openenv.yaml](/Users/ronoktanvir/Documents/openenv-hacks/openenv.yaml)
- [server/Dockerfile](/Users/ronoktanvir/Documents/openenv-hacks/server/Dockerfile)

Deploy with:

```bash
UV_CACHE_DIR=/tmp/uv-cache ./.venv/bin/openenv push . --repo-id <hf-username>/overseer-openenv
```

After deployment, add the Space secret:

- `ANTHROPIC_API_KEY`
