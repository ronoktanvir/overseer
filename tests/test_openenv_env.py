import unittest

from models import OverseerAction
from server.overseer_environment import OverseerEnvironment


def make_sample(target_player: str, turn: str, true_strategy: str):
    return {
        "observation": {
            "target_player": target_player,
            "turn": turn,
            "current_state": {
                "turn": turn,
                "units": {target_player: ["A PAR"]},
                "supply_centers": {target_player: ["PAR"]},
                "orders": {target_player: ["A PAR H"]},
                "conflicts": {},
            },
            "history": [],
            "communications": {target_player: {"messaged": [], "ignored": [], "message_count": 0}},
            "communication_shift": {},
            "public_chat": [],
        },
        "true_strategy": true_strategy,
    }


class OverseerEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_reset_and_step_async(self):
        env = OverseerEnvironment(
            samples=[
                make_sample("FRANCE", "S1905M", "Take Belgium while calming England."),
                make_sample("GERMANY", "F1905M", "Defend Munich and watch Russia."),
            ],
            judge_fn=lambda target, truth, prediction: float("Belgium" in prediction),
        )

        reset_obs = env.reset()
        self.assertEqual(reset_obs.target_player, "FRANCE")
        self.assertEqual(env.state.index, 0)

        next_obs = await env.step_async(
            OverseerAction(target_player="FRANCE", prediction="France wants Belgium.")
        )
        self.assertEqual(next_obs.target_player, "GERMANY")
        self.assertEqual(next_obs.reward, 1.0)
        self.assertFalse(next_obs.done)
        self.assertEqual(env.state.index, 1)

    async def test_terminal_step(self):
        env = OverseerEnvironment(
            samples=[make_sample("TURKEY", "S1906M", "Hold and consolidate.")],
            judge_fn=lambda target, truth, prediction: 0.0,
        )
        env.reset()

        terminal_obs = await env.step_async(
            OverseerAction(target_player="TURKEY", prediction="Attack now.")
        )

        self.assertTrue(terminal_obs.done)
        self.assertEqual(terminal_obs.reward, 0.0)
        self.assertEqual(env.state.index, 1)
        self.assertIsNone(env.state.current_target_player)


if __name__ == "__main__":
    unittest.main()
