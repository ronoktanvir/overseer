import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from judge import StrategyJudge
from training.minimal_trl_sft import (
    DEFAULT_MODEL,
    build_prompt,
    evaluate_model,
    keep_latest_train_games,
    load_samples,
    maybe_cap_samples,
    split_samples_by_game,
    token_f1,
)

DEFAULT_OUTPUT_DIR = "outputs/minimal_trl_grpo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HF TRL GRPO training script for the Diplomacy overseer.")
    parser.add_argument("--data-path", default="game_data.json", help="Path to repaired training data JSON.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base model to fine-tune.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for adapters and metrics.")
    parser.add_argument(
        "--adapter-path",
        default="",
        help="Optional existing LoRA adapter directory to continue training from.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--eval-games", type=int, default=2, help="How many whole games to hold out.")
    parser.add_argument(
        "--latest-train-games",
        type=int,
        default=0,
        help="If set, train only on the newest N train games immediately before the eval holdout.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on training prompts.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--max-steps", type=int, default=20, help="Number of GRPO optimization steps.")
    parser.add_argument("--max-eval-samples", type=int, default=32, help="Held-out samples to score before/after.")
    parser.add_argument("--num-generations", type=int, default=2, help="Completions sampled per prompt.")
    parser.add_argument("--max-prompt-length", type=int, default=2304, help="Maximum prompt length.")
    parser.add_argument("--max-completion-length", type=int, default=96, help="Maximum completion length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for GRPO rollouts.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for rollout sampling.")
    parser.add_argument(
        "--reward-mode",
        choices=["judge", "token_f1"],
        default="judge",
        help="Reward function for GRPO. Use token_f1 if Anthropic auth is unavailable.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading. Use this if bitsandbytes is unavailable.",
    )
    return parser.parse_args()


def make_grpo_dataset(samples: list[dict[str, Any]]) -> Dataset:
    rows = []
    for sample in samples:
        observation = sample["observation"]
        rows.append(
            {
                "prompt": build_prompt(observation),
                "true_strategy": sample["true_strategy"].strip(),
                "target_player": observation.get("target_player", ""),
                "game_id": observation.get("game_id", 0),
                "turn": observation.get("turn", ""),
            }
        )
    return Dataset.from_list(rows)


def load_grpo_model_and_tokenizer(model_name: str, use_4bit: bool):
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


def _flatten_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        pieces = []
        for item in completion:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    pieces.append(str(text))
            else:
                pieces.append(str(item))
        return "".join(pieces).strip()
    return str(completion).strip()


def build_judge_reward_func():
    judge = StrategyJudge()

    if not judge.client:
        raise RuntimeError("ANTHROPIC_API_KEY is required for GRPO judge rewards.")

    async def _score_batch(
        completions: list[str],
        true_strategy: list[str],
        target_player: list[str],
    ) -> list[float]:
        tasks = []
        for prediction, truth, power in zip(completions, true_strategy, target_player):
            prediction = prediction.strip()
            if not prediction:
                tasks.append(asyncio.sleep(0, result=0))
            else:
                tasks.append(
                    judge.score_prediction(
                        target_player=power,
                        true_strategy=truth,
                        predicted_strategy=prediction,
                    )
                )
        scores = await asyncio.gather(*tasks)
        return [float(score) for score in scores]

    def judge_reward_func(completions, true_strategy, target_player, **kwargs) -> list[float]:
        flattened = [_flatten_completion(completion) for completion in completions]
        return asyncio.run(_score_batch(flattened, list(true_strategy), list(target_player)))

    judge_reward_func.__name__ = "judge_binary_reward"
    return judge_reward_func


def build_token_f1_reward_func():
    def token_f1_reward_func(completions, true_strategy, **kwargs) -> list[float]:
        flattened = [_flatten_completion(completion) for completion in completions]
        return [float(token_f1(prediction, truth)) for prediction, truth in zip(flattened, true_strategy)]

    token_f1_reward_func.__name__ = "token_f1_reward"
    return token_f1_reward_func


def summarize_log_history(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    reward_keys = []
    for entry in log_history:
        for key in entry:
            if "reward" in key.lower():
                reward_keys.append(key)
    reward_keys = sorted(set(reward_keys))

    reward_history = []
    for entry in log_history:
        reward_entry = {"step": entry.get("step")}
        found = False
        for key in reward_keys:
            if key in entry:
                reward_entry[key] = entry[key]
                found = True
        if found:
            reward_history.append(reward_entry)

    return {
        "reward_keys": reward_keys,
        "reward_history": reward_history,
        "log_history": log_history,
    }


def save_metrics(output_dir: str, metrics: dict[str, Any]) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)


def build_grpo_config(args: argparse.Namespace, *, bf16: bool, fp16: bool) -> GRPOConfig:
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "bf16": False,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    supported = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return GRPOConfig(**filtered)


def main() -> None:
    args = parse_args()

    samples = load_samples(args.data_path)
    train_samples, eval_samples = split_samples_by_game(samples, args.eval_games)
    train_samples = keep_latest_train_games(train_samples, args.latest_train_games)
    train_samples = maybe_cap_samples(train_samples, args.max_train_samples)

    eval_samples = maybe_cap_samples(eval_samples, args.max_eval_samples)

    if not train_samples:
        raise ValueError("No training samples available. Check game_id annotations in game_data.json.")

    model, tokenizer = load_grpo_model_and_tokenizer(args.model_name, use_4bit=not args.no_4bit)

    train_dataset = make_grpo_dataset(train_samples)
    if args.reward_mode == "judge":
        reward_func = build_judge_reward_func()
    else:
        reward_func = build_token_f1_reward_func()

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)
    model.warnings_issued = {}

    bf16 = False
    fp16 = torch.cuda.is_available()

    pre_metrics = None
    if eval_samples:
        pre_metrics = evaluate_model(
            model,
            tokenizer,
            eval_samples,
            max_new_tokens=args.max_completion_length,
            max_prompt_length=args.max_prompt_length,
            use_judge=args.reward_mode == "judge",
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=build_grpo_config(args, bf16=bf16, fp16=fp16),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    post_metrics = None
    if eval_samples:
        post_metrics = evaluate_model(
            trainer.model,
            tokenizer,
            eval_samples,
            max_new_tokens=args.max_completion_length,
            max_prompt_length=args.max_prompt_length,
            use_judge=args.reward_mode == "judge",
        )

    summary = summarize_log_history(trainer.state.log_history)
    metrics = {
        "model_name": args.model_name,
        "train_samples": len(train_samples),
        "train_games": len({sample["observation"]["game_id"] for sample in train_samples}),
        "held_out_games": len({sample["observation"]["game_id"] for sample in eval_samples}),
        "eval_samples": len(eval_samples),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "reward_mode": args.reward_mode,
        "used_binary_judge_reward": args.reward_mode == "judge",
        "adapter_path": args.adapter_path or None,
        "pre_eval": pre_metrics,
        "post_eval": post_metrics,
        "eval_improvement": (
            (post_metrics["metric_value"] - pre_metrics["metric_value"])
            if pre_metrics and post_metrics
            else None
        ),
        **summary,
    }
    save_metrics(args.output_dir, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
