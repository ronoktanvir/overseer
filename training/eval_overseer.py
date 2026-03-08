import argparse
import json
import os
from pathlib import Path

from peft import PeftModel

from training.minimal_trl_sft import (
    DEFAULT_MODEL,
    evaluate_model,
    load_model_and_tokenizer,
    load_samples,
    maybe_cap_samples,
    split_samples_by_game,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base model or adapter on held-out overseer samples.")
    parser.add_argument("--data-path", default="game_data.json", help="Path to repaired training data JSON.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base model to evaluate.")
    parser.add_argument("--adapter-path", default="", help="Optional PEFT adapter directory to load on top of the base model.")
    parser.add_argument("--eval-games", type=int, default=1, help="How many whole games to hold out for evaluation.")
    parser.add_argument("--max-eval-samples", type=int, default=32, help="How many eval samples to score.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation length for evaluation.")
    parser.add_argument("--max-prompt-length", type=int, default=3072, help="Maximum prompt length during evaluation.")
    parser.add_argument(
        "--judge-eval",
        action="store_true",
        help="Use the Anthropic binary judge for evaluation. Requires ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading. Use this if bitsandbytes is unavailable.",
    )
    parser.add_argument("--output-path", default="", help="Optional path to write metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_samples(args.data_path)
    _, eval_samples = split_samples_by_game(samples, args.eval_games)
    eval_samples = maybe_cap_samples(eval_samples, args.max_eval_samples)
    if not eval_samples:
        raise ValueError("No eval samples available.")

    use_judge = args.judge_eval and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if args.judge_eval and not use_judge:
        print("ANTHROPIC_API_KEY not found; falling back to token_f1 evaluation.")

    model, tokenizer = load_model_and_tokenizer(args.model_name, use_4bit=not args.no_4bit)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    metrics = evaluate_model(
        model,
        tokenizer,
        eval_samples,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        use_judge=use_judge,
    )
    payload = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path or None,
        "eval_games": len({sample["observation"]["game_id"] for sample in eval_samples}),
        "eval_samples": len(eval_samples),
        "used_judge_eval": use_judge,
        **metrics,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
