import os

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from openenv.core.env_server.http_server import create_app, create_fastapi_app
    from openenv.core.env_server.web_interface import (
        WebInterfaceManager,
        get_web_interface_html,
        load_environment_metadata,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core[core]==0.2.1 is required. Install dependencies before launching the server."
    ) from e

try:
    from models import OverseerAction, OverseerObservation
except ImportError:  # pragma: no cover
    from overseer_env.models import OverseerAction, OverseerObservation

from .overseer_environment import OverseerEnvironment


class AutoResetWebInterfaceManager(WebInterfaceManager):
    """Seed the built-in OpenEnv UI with an initial observation on first connect."""

    async def connect_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.connected_clients.append(websocket)

        if self.episode_state.current_observation is None:
            try:
                await self.reset_environment()
                return
            except Exception:
                # Fall back to the default blank state if reset fails.
                pass

        await self._send_state_update()


def create_overseer_app():
    enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
    if not enable_web:
        return create_app(
            OverseerEnvironment,
            OverseerAction,
            OverseerObservation,
            env_name="overseer",
            max_concurrent_envs=8,
        )

    app = create_fastapi_app(
        OverseerEnvironment,
        OverseerAction,
        OverseerObservation,
        max_concurrent_envs=8,
    )
    metadata = load_environment_metadata(OverseerEnvironment, "overseer")
    web_manager = AutoResetWebInterfaceManager(
        OverseerEnvironment,
        OverseerAction,
        OverseerObservation,
        metadata,
    )

    @app.get("/web", response_class=HTMLResponse)
    async def web_interface():
        return get_web_interface_html(OverseerAction, web_manager.metadata)

    @app.get("/web/metadata")
    async def web_metadata():
        return web_manager.metadata.model_dump()

    @app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset():
        return await web_manager.reset_environment()

    @app.post("/web/step")
    async def web_step(request: dict):
        if "message" in request:
            message = request["message"]
            action = web_manager.env.message_to_action(message)
            action_data = {"tokens": action.tokens.tolist()}
        else:
            action_data = request.get("action", {})

        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        return web_manager.get_state()

    return app


app = create_overseer_app()


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
