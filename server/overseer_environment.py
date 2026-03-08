import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from judge import judge_prediction
    from models import OverseerAction, OverseerObservation, OverseerState
except ImportError:  # pragma: no cover
    from overseer_env.judge import judge_prediction
    from overseer_env.models import OverseerAction, OverseerObservation, OverseerState

DEFAULT_DATA_PATH = "game_data.json"


def _load_samples(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, "r") as f:
        return json.load(f)["training_data"]


class OverseerEnvironment(Environment[OverseerAction, OverseerObservation, OverseerState]):
    """OpenEnv environment that serves pre-generated Diplomacy overseer samples."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        samples: Optional[list[dict[str, Any]]] = None,
        judge_fn: Optional[Callable[[str, str, str], Any]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self._samples = samples if samples is not None else _load_samples(data_path)
        self._judge_fn = judge_fn or judge_prediction
        self._state = OverseerState(
            episode_id=str(uuid4()),
            step_count=0,
            index=0,
            total_samples=len(self._samples),
            current_target_player=None,
            data_path=data_path,
        )

    def _sample_to_observation(
        self,
        observation_dict: dict[str, Any],
        reward: float | None = None,
        done: bool = False,
    ) -> OverseerObservation:
        return OverseerObservation(**observation_dict, reward=reward, done=done)

    def _terminal_observation(self, reward: float) -> OverseerObservation:
        return OverseerObservation(
            reward=reward,
            done=True,
            metadata={"status": "completed", "data_path": self.data_path},
        )

    def _current_sample(self) -> dict[str, Any]:
        if not self._samples:
            raise RuntimeError(f"No training_data found in {self.data_path}")
        if self._state.index >= len(self._samples):
            raise RuntimeError("Environment is done. Call reset() before stepping again.")
        return self._samples[self._state.index]

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OverseerObservation:
        self._state = OverseerState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            index=0,
            total_samples=len(self._samples),
            current_target_player=None,
            data_path=self.data_path,
        )
        sample = self._current_sample()
        observation = self._sample_to_observation(sample["observation"])
        self._state.current_target_player = observation.target_player
        return observation

    async def step_async(
        self,
        action: OverseerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OverseerObservation:
        sample = self._current_sample()
        expected_target = sample["observation"]["target_player"]

        if action.target_player and action.target_player.upper() != expected_target:
            raise ValueError(
                f"Action target_player={action.target_player!r} does not match current observation target "
                f"{expected_target!r}."
            )

        judge_result = self._judge_fn(expected_target, sample["true_strategy"], action.prediction)
        if asyncio.iscoroutine(judge_result):
            reward = float(await judge_result)
        else:
            reward = float(judge_result)

        self._state.index += 1
        self._state.step_count += 1

        done = self._state.index >= len(self._samples)
        if done:
            self._state.current_target_player = None
            return self._terminal_observation(reward)

        next_sample = self._samples[self._state.index]
        next_observation = self._sample_to_observation(next_sample["observation"], reward=reward, done=False)
        self._state.current_target_player = next_observation.target_player
        return next_observation

    def step(
        self,
        action: OverseerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OverseerObservation:
        return asyncio.run(self.step_async(action, timeout_s=timeout_s, **kwargs))

    @property
    def state(self) -> OverseerState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Diplomacy Overseer",
            description="Predict hidden Diplomacy player strategies from public board and communication signals.",
            version="0.2.1",
            author="openenv-hacks",
        )
