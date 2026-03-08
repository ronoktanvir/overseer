from typing import Any

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class OverseerAction(Action):
    """Action sent to the OpenEnv server for one overseer prediction."""

    prediction: str = Field(..., description="Predicted hidden strategy for the current target player.")
    target_player: str | None = Field(
        default=None,
        description="Optional target power. If provided, it must match the current observation.",
    )


class OverseerObservation(Observation):
    """Observation exposed to the overseer model."""

    target_player: str = Field(default="", description="Power whose hidden strategy should be predicted.")
    turn: str = Field(default="", description="Current game phase.")
    current_state: dict[str, Any] = Field(default_factory=dict, description="Current board snapshot.")
    history: list[dict[str, Any]] = Field(default_factory=list, description="Last few observed turns.")
    communications: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-power communication metadata visible to the overseer.",
    )
    communication_shift: dict[str, Any] = Field(
        default_factory=dict,
        description="Turn-over-turn message-count deltas.",
    )
    public_chat: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Last few turns of public broadcasts.",
    )


class OverseerState(State):
    """State tracked by the OpenEnv server between requests."""

    index: int = Field(default=0, ge=0, description="Current sample index.")
    total_samples: int = Field(default=0, ge=0, description="Total number of available samples.")
    current_target_player: str | None = Field(
        default=None,
        description="Target player for the current sample.",
    )
    data_path: str = Field(default="game_data.json", description="Dataset backing this environment.")
