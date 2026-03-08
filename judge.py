import os
from typing import Mapping

import anthropic

from prompts import JUDGE_PROMPT

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def parse_binary_score(text: str) -> int:
    """Parse a strict 0/1 judge response."""
    normalized = (text or "").strip()
    if normalized not in {"0", "1"}:
        raise ValueError(f"Judge must return only '0' or '1', got {normalized!r}")
    return int(normalized)


class StrategyJudge:
    """Binary similarity judge for overseer predictions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JUDGE_MODEL,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = client or (
            anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        )

    async def score_prediction(
        self,
        target_player: str,
        true_strategy: str,
        predicted_strategy: str,
    ) -> int:
        if not self.client:
            raise RuntimeError("ANTHROPIC_API_KEY is required to score predictions.")

        prompt = JUDGE_PROMPT.format(
            power=target_player,
            true_strategy=true_strategy,
            predicted_strategy=predicted_strategy,
        )
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_binary_score(response.content[0].text)

    async def score_turn(
        self,
        predictions: Mapping[str, str],
        true_strategies: Mapping[str, str],
    ) -> dict[str, object]:
        """Score one full Diplomacy turn and return per-player scores plus the average."""
        scores: dict[str, int] = {}
        for power, true_strategy in true_strategies.items():
            if power not in predictions:
                raise ValueError(f"Missing prediction for {power}")
            scores[power] = await self.score_prediction(
                target_player=power,
                true_strategy=true_strategy,
                predicted_strategy=predictions[power],
            )

        average_reward = sum(scores.values()) / len(scores) if scores else 0.0
        return {"scores": scores, "average_reward": average_reward}


async def judge_prediction(
    target_player: str,
    true_strategy: str,
    predicted_strategy: str,
) -> float:
    judge = StrategyJudge()
    return float(
        await judge.score_prediction(
            target_player=target_player,
            true_strategy=true_strategy,
            predicted_strategy=predicted_strategy,
        )
    )
