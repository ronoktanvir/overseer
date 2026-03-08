try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core[core]==0.2.1 is required. Install dependencies before launching the server."
    ) from e

try:
    from models import OverseerAction, OverseerObservation
except ImportError:  # pragma: no cover
    from overseer_env.models import OverseerAction, OverseerObservation

from .overseer_environment import OverseerEnvironment


app = create_app(
    OverseerEnvironment,
    OverseerAction,
    OverseerObservation,
    env_name="overseer",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
