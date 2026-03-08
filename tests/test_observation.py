import unittest

from diplomacy import Game

from observation import POWERS, build_current_state


class BuildCurrentStateTests(unittest.TestCase):
    def test_snapshots_explicit_submitted_orders(self):
        game = Game()
        submitted_orders = {power: [] for power in POWERS}
        submitted_orders["ENGLAND"] = ["F EDI - NWG", "F LON - NTH", "A LVP - YOR"]

        state = build_current_state(
            game,
            phase=game.get_current_phase(),
            submitted_orders=submitted_orders,
        )

        self.assertEqual(state["orders"]["ENGLAND"], submitted_orders["ENGLAND"])
        self.assertEqual(state["orders"]["FRANCE"], [])

    def test_falls_back_to_game_orders_when_snapshot_missing(self):
        game = Game()
        game.set_orders("ENGLAND", ["F EDI - NWG", "F LON - NTH", "A LVP - YOR"])

        state = build_current_state(game, phase=game.get_current_phase())

        self.assertEqual(state["orders"]["ENGLAND"], game.get_orders("ENGLAND"))


if __name__ == "__main__":
    unittest.main()
