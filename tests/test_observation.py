import unittest

from diplomacy import Game

from observation import POWERS, build_current_state, build_observation


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

    def test_includes_communication_metadata_when_tracker_present(self):
        game = Game()
        comm_tracker = {
            "curr": {
                power: {"messaged": [], "ignored": [], "message_count": 0}
                for power in POWERS
            },
            "prev": {
                power: {"messaged": [], "ignored": [], "message_count": 0}
                for power in POWERS
            },
        }
        comm_tracker["curr"]["AUSTRIA"] = {
            "messaged": ["ITALY"],
            "ignored": ["RUSSIA"],
            "message_count": 1,
        }

        state = build_current_state(
            game,
            phase=game.get_current_phase(),
            comm_tracker=comm_tracker,
        )

        self.assertEqual(state["communications"]["AUSTRIA"]["messaged"], ["ITALY"])
        self.assertEqual(state["communication_shift"]["AUSTRIA"], 1)


class BuildObservationTests(unittest.TestCase):
    def test_includes_target_comm_and_current_state_turn(self):
        current_state = {
            "turn": "S1902M",
            "units": {"AUSTRIA": ["A VIE"]},
            "supply_centers": {"AUSTRIA": ["VIE"]},
            "orders": {"AUSTRIA": ["A VIE H"]},
            "conflicts": {},
        }
        comm_tracker = {
            "curr": {
                "AUSTRIA": {"messaged": ["ITALY"], "ignored": ["RUSSIA"], "message_count": 1},
                "ENGLAND": {"messaged": [], "ignored": [], "message_count": 0},
                "FRANCE": {"messaged": [], "ignored": [], "message_count": 0},
                "GERMANY": {"messaged": [], "ignored": [], "message_count": 0},
                "ITALY": {"messaged": [], "ignored": [], "message_count": 0},
                "RUSSIA": {"messaged": [], "ignored": [], "message_count": 0},
                "TURKEY": {"messaged": [], "ignored": [], "message_count": 0},
            },
            "prev": {},
        }

        observation = build_observation("AUSTRIA", current_state, [], comm_tracker, [])

        self.assertIn("AUSTRIA", observation["communications"])
        self.assertEqual(observation["communications"]["AUSTRIA"]["messaged"], ["ITALY"])
        self.assertEqual(observation["current_state"]["turn"], "S1902M")

    def test_snapshots_public_chat(self):
        current_state = {
            "turn": "S1902M",
            "units": {"AUSTRIA": ["A VIE"]},
            "supply_centers": {"AUSTRIA": ["VIE"]},
            "orders": {"AUSTRIA": ["A VIE H"]},
            "conflicts": {},
        }
        public_chat = [{"turn": "S1902M", "messages": [["AUSTRIA", "hello"]]}]

        observation = build_observation("AUSTRIA", current_state, [], {"curr": {}, "prev": {}}, public_chat)
        public_chat.append({"turn": "F1902M", "messages": [["ENGLAND", "future"]]})

        self.assertEqual(len(observation["public_chat"]), 1)
        self.assertEqual(observation["public_chat"][0]["turn"], "S1902M")

    def test_history_preserves_communication_metadata(self):
        history_turn = {
            "turn": "S1902M",
            "units": {"AUSTRIA": ["A VIE"]},
            "supply_centers": {"AUSTRIA": ["VIE"]},
            "orders": {"AUSTRIA": ["A VIE H"]},
            "conflicts": {},
            "communications": {
                "AUSTRIA": {"messaged": ["ITALY"], "ignored": ["RUSSIA"], "message_count": 1}
            },
            "communication_shift": {"AUSTRIA": 1},
        }

        observation = build_observation(
            "AUSTRIA",
            history_turn,
            [history_turn],
            {"curr": {}, "prev": {}},
            [],
        )

        self.assertEqual(
            observation["history"][0]["communications"]["AUSTRIA"]["messaged"],
            ["ITALY"],
        )
        self.assertEqual(observation["history"][0]["communication_shift"]["AUSTRIA"], 1)


if __name__ == "__main__":
    unittest.main()
