# Diplomacy Overseer

OpenEnv environment and training harness for an overseer model that infers hidden Diplomacy strategy from public state, order history, communication metadata, and public messages.

This repository contains:

- an OpenEnv-compatible Diplomacy overseer environment
- local tooling for repairing serialized game data
- training and evaluation scripts for SFT and GRPO
- a lightweight demo dashboard for inspecting overseer predictions

Tracks:
- Statement 1: Multi-Agent Interactions
- Fleet AI bonus: Scalable Oversight

## Repo Layout

- [`server/app.py`](server/app.py): OpenEnv HTTP server entrypoint
- [`server/overseer_environment.py`](server/overseer_environment.py): offline overseer environment over local game data
- [`judge.py`](judge.py): binary strategy-similarity judge
- [`repair_game_data.py`](repair_game_data.py): repair and annotate raw serialized game data
- [`overseer_server.py`](overseer_server.py): local dashboard for visualizing per-turn overseer predictions
- [`training/minimal_trl_sft.py`](training/minimal_trl_sft.py): supervised fine-tuning script
- [`training/minimal_trl_grpo.py`](training/minimal_trl_grpo.py): GRPO training loop using the judge as reward
- [`training/eval_overseer.py`](training/eval_overseer.py): evaluation helper for base models or adapters
- [`training/export_metric_csv.py`](training/export_metric_csv.py): metrics export helper
- [`colab/minimal_trl_overseer.ipynb`](colab/minimal_trl_overseer.ipynb): Colab wrapper for training

## Quickstart

Install dependencies:

```bash
python -m venv .venv
./.venv/bin/pip install -e ".[dev]"
```

Prepare local data:

```bash
python repair_game_data.py path/to/source.json \
  --output game_data.json \
  --public-chat-window 5
```

`game_data*.json` files are intentionally ignored and kept local. Every script accepts `--data-path` if you prefer another location.

## Run The OpenEnv Server

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

`POST /step` requires `ANTHROPIC_API_KEY` because reward scoring uses the binary judge.

## Demo Dashboard

```bash
./.venv/bin/uvicorn overseer_server:app --host 0.0.0.0 --port 8002
```

Open `http://localhost:8002`.

This dashboard is only for visualization. It is not the OpenEnv server used for training.

## Training And Evaluation

Supervised fine-tuning:

```bash
python training/minimal_trl_sft.py \
  --data-path game_data.json \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/minimal_trl_sft \
  --num-train-epochs 1 \
  --max-eval-samples 8 \
  --judge-eval
```

GRPO training:

```bash
python -m training.minimal_trl_grpo \
  --data-path game_data.json \
  --output-dir outputs/minimal_trl_grpo \
  --max-steps 20 \
  --num-generations 2
```

Evaluation:

```bash
python -m training.eval_overseer \
  --data-path game_data.json \
  --adapter-path outputs/minimal_trl_sft \
  --judge-eval
```

Each observation is expected to include:

- `game_id`
- `game_step_index`
- `turn`
- `target_player`
- `current_state`
- `history`
- `communications`
- `communication_shift`
- `public_chat`

## Tests

```bash
./.venv/bin/python -m pytest
```

## Deployment

This repo is packaged for OpenEnv / Hugging Face Spaces deployment with [`openenv.yaml`](openenv.yaml) and [`server/Dockerfile`](server/Dockerfile).

```bash
UV_CACHE_DIR=/tmp/uv-cache ./.venv/bin/openenv push . --repo-id <hf-username>/overseer-openenv
```

Add `ANTHROPIC_API_KEY` as a Space secret after deployment.
