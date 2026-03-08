from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import OverseerAction, OverseerObservation, OverseerState


class OverseerEnv(EnvClient[OverseerAction, OverseerObservation, OverseerState]):
    """Typed client for the Diplomacy overseer OpenEnv server."""

    def _step_payload(self, action: OverseerAction) -> dict[str, Any]:
        payload: dict[str, Any] = {"prediction": action.prediction}
        if action.target_player is not None:
            payload["target_player"] = action.target_player
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[OverseerObservation]:
        observation = OverseerObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> OverseerState:
        return OverseerState(**payload)
