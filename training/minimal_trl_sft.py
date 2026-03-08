import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from judge import StrategyJudge

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/minimal_trl_sft"
PROMPT_HEADER = (
    "You are a Diplomacy strategy overseer. Infer the target player's hidden strategy from the "
    "observable board state, short history, communication metadata, and public chat. "
    "Respond in 1-3 sentences."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HF TRL training script for the Diplomacy overseer task.")
    parser.add_argument("--data-path", default="game_data.json", help="Path to the repaired training data JSON.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base model to fine-tune.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoints and metrics.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--eval-games", type=int, default=2, help="How many whole games to hold out for evaluation.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on training samples.")
    parser.add_argument("--max-eval-samples", type=int, default=16, help="How many eval samples to score before/after.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum tokenized sequence length.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2, help="Train batch size per device.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of train epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation length for evaluation.")
    parser.add_argument(
        "--judge-eval",
        action="store_true",
        help="Use the Anthropic binary judge for before/after evaluation. Requires ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading. Use this if bitsandbytes is unavailable or you want full precision.",
    )
    return parser.parse_args()


def load_samples(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, "r") as handle:
        payload = json.load(handle)
    return payload["training_data"]


def compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields that matter for overseer training while staying readable in a prompt."""
    return {
        "game_id": observation.get("game_id"),
        "game_step_index": observation.get("game_step_index"),
        "target_player": observation.get("target_player"),
        "turn": observation.get("turn"),
        "current_state": observation.get("current_state", {}),
        "history": observation.get("history", []),
        "communications": observation.get("communications", {}),
        "communication_shift": observation.get("communication_shift", {}),
        "public_chat": observation.get("public_chat", []),
    }


def build_prompt(observation: dict[str, Any]) -> str:
    serialized = json.dumps(compact_observation(observation), indent=2)
    return (
        f"{PROMPT_HEADER}\n\n"
        f"Observation:\n{serialized}\n\n"
        "Predicted hidden strategy:\n"
    )


def build_training_text(sample: dict[str, Any]) -> str:
    return build_prompt(sample["observation"]) + sample["true_strategy"].strip()


def split_samples_by_game(samples: list[dict[str, Any]], eval_games: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    game_ids = sorted({sample["observation"].get("game_id", 0) for sample in samples})
    if len(game_ids) <= 1:
        cutoff = max(1, int(len(samples) * 0.1))
        return samples[:-cutoff], samples[-cutoff:]

    eval_games = max(1, min(eval_games, len(game_ids) - 1))
    eval_game_ids = set(game_ids[-eval_games:])
    train_samples = [sample for sample in samples if sample["observation"].get("game_id", 0) not in eval_game_ids]
    eval_samples = [sample for sample in samples if sample["observation"].get("game_id", 0) in eval_game_ids]
    return train_samples, eval_samples


def maybe_cap_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit and len(samples) > limit:
        return samples[:limit]
    return samples


def make_dataset(samples: list[dict[str, Any]]) -> Dataset:
    rows = [{"text": build_training_text(sample)} for sample in samples]
    return Dataset.from_list(rows)


def load_model_and_tokenizer(model_name: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_kwargs = {"device_map": "auto"}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    return model, tokenizer


def decode_completion(tokenizer, prompt: str, generated_ids) -> str:
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    prompt_tail = re.escape(prompt[-80:])
    match = re.search(prompt_tail + r"(.*)$", full_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return full_text.strip()


@torch.inference_mode()
def generate_prediction(model, tokenizer, observation: dict[str, Any], max_new_tokens: int) -> str:
    prompt = build_prompt(observation)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return decode_completion(tokenizer, prompt, generated_ids)


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = re.findall(r"\w+", prediction.lower())
    target_tokens = re.findall(r"\w+", target.lower())
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    target_counts: dict[str, int] = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, target_counts.get(token, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


async def judge_mean_reward(samples: list[dict[str, Any]], predictions: list[str]) -> float:
    judge = StrategyJudge()
    scores = []
    for sample, prediction in zip(samples, predictions):
        score = await judge.score_prediction(
            target_player=sample["observation"]["target_player"],
            true_strategy=sample["true_strategy"],
            predicted_strategy=prediction,
        )
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_model(model, tokenizer, samples: list[dict[str, Any]], max_new_tokens: int, use_judge: bool) -> dict[str, Any]:
    subset = samples
    predictions = [
        generate_prediction(model, tokenizer, sample["observation"], max_new_tokens=max_new_tokens)
        for sample in subset
    ]

    if use_judge:
        metric_name = "judge_reward"
        metric_value = asyncio.run(judge_mean_reward(subset, predictions))
    else:
        metric_name = "token_f1"
        metric_value = sum(
            token_f1(prediction, sample["true_strategy"])
            for sample, prediction in zip(subset, predictions)
        ) / len(subset)

    preview = []
    for sample, prediction in zip(subset[:3], predictions[:3]):
        preview.append(
            {
                "game_id": sample["observation"].get("game_id"),
                "turn": sample["observation"].get("turn"),
                "target_player": sample["observation"].get("target_player"),
                "prediction": prediction,
                "true_strategy": sample["true_strategy"],
            }
        )

    return {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "preview": preview,
    }


def save_metrics(output_dir: str, metrics: dict[str, Any]) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    samples = load_samples(args.data_path)
    train_samples, eval_samples = split_samples_by_game(samples, args.eval_games)
    train_samples = maybe_cap_samples(train_samples, args.max_train_samples)
    eval_samples = maybe_cap_samples(eval_samples, args.max_eval_samples)

    if not train_samples or not eval_samples:
        raise ValueError("Need non-empty train and eval splits. Check game_id annotations in game_data.json.")

    use_judge = args.judge_eval and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if args.judge_eval and not use_judge:
        print("ANTHROPIC_API_KEY not found; falling back to token_f1 evaluation.")

    model, tokenizer = load_model_and_tokenizer(args.model_name, use_4bit=not args.no_4bit)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    train_dataset = make_dataset(train_samples)
    eval_dataset = make_dataset(eval_samples)

    pre_metrics = evaluate_model(
        model,
        tokenizer,
        eval_samples,
        max_new_tokens=args.max_new_tokens,
        use_judge=use_judge,
    )
    print(f"Pre-train {pre_metrics['metric_name']}: {pre_metrics['metric_value']:.4f}")

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = torch.cuda.is_available() and not bf16
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=5,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            max_length=args.max_seq_length,
            bf16=bf16,
            fp16=fp16,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=lambda example: example["text"],
        peft_config=peft_config,
    )
    trainer.train()

    post_metrics = evaluate_model(
        trainer.model,
        tokenizer,
        eval_samples,
        max_new_tokens=args.max_new_tokens,
        use_judge=use_judge,
    )
    print(f"Post-train {post_metrics['metric_name']}: {post_metrics['metric_value']:.4f}")

    metrics = {
        "model_name": args.model_name,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "train_games": len({sample['observation']['game_id'] for sample in train_samples}),
        "eval_games": len({sample['observation']['game_id'] for sample in eval_samples}),
        "pre": pre_metrics,
        "post": post_metrics,
        "improvement": post_metrics["metric_value"] - pre_metrics["metric_value"],
        "used_judge_eval": use_judge,
    }
    save_metrics(args.output_dir, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
