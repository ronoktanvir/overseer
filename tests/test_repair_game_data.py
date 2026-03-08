import unittest

from repair_game_data import repair_game_data


def sample(turn, target, public_chat, communications, communication_shift=None, history=None):
    return {
        "observation": {
            "target_player": target,
            "turn": turn,
            "current_state": {"orders": {}, "units": {}, "supply_centers": {}, "conflicts": {}},
            "history": history or [],
            "communications": communications,
            "communication_shift": communication_shift or {},
            "public_chat": public_chat,
        },
        "true_strategy": "example",
    }


class RepairGameDataTests(unittest.TestCase):
    def test_repairs_public_chat_and_target_communications(self):
        full_public_chat = [
            {"turn": "S1902M", "messages": [["AUSTRIA", "one"]]},
            {"turn": "F1902M", "messages": [["ENGLAND", "two"]]},
            {"turn": "W1902A", "messages": [["FRANCE", "three"]]},
        ]
        data = {
            "training_data": [
                sample(
                    "S1902M",
                    "AUSTRIA",
                    full_public_chat,
                    {"ENGLAND": {"messaged": ["FRANCE"], "ignored": [], "message_count": 1}},
                ),
                sample(
                    "S1902M",
                    "ENGLAND",
                    full_public_chat,
                    {"AUSTRIA": {"messaged": ["ITALY"], "ignored": [], "message_count": 1}},
                ),
                sample(
                    "F1902M",
                    "AUSTRIA",
                    full_public_chat,
                    {"ENGLAND": {"messaged": ["FRANCE"], "ignored": [], "message_count": 1}},
                ),
                sample(
                    "F1902M",
                    "ENGLAND",
                    full_public_chat,
                    {"AUSTRIA": {"messaged": ["ITALY"], "ignored": [], "message_count": 1}},
                ),
            ]
        }

        repaired, stats = repair_game_data(data, public_chat_window=2)

        first_obs = repaired["training_data"][0]["observation"]
        self.assertEqual([entry["turn"] for entry in first_obs["public_chat"]], ["S1902M"])
        self.assertIn("AUSTRIA", first_obs["communications"])
        self.assertEqual(first_obs["current_state"]["turn"], "S1902M")

        second_turn_obs = repaired["training_data"][2]["observation"]
        self.assertEqual([entry["turn"] for entry in second_turn_obs["public_chat"]], ["S1902M", "F1902M"])

        self.assertEqual(stats["public_chat_trimmed"], 4)
        self.assertEqual(stats["target_communications_backfilled"], 4)
        self.assertEqual(stats["current_state_turn_added"], 4)
        self.assertEqual(stats["unresolved_target_communications"], [])

    def test_backfills_history_communications_by_game(self):
        full_public_chat = [{"turn": "S1902M", "messages": []}, {"turn": "F1902M", "messages": []}]
        data = {
            "training_data": [
                sample(
                    "S1902M",
                    "AUSTRIA",
                    full_public_chat,
                    {"AUSTRIA": {"messaged": ["ITALY"], "ignored": [], "message_count": 1}},
                    {"AUSTRIA": 1},
                    history=[{"turn": "S1901M"}, {"turn": "S1902M"}],
                ),
                sample(
                    "F1902M",
                    "AUSTRIA",
                    full_public_chat,
                    {"AUSTRIA": {"messaged": ["GERMANY"], "ignored": [], "message_count": 1}},
                    {"AUSTRIA": -1},
                    history=[{"turn": "S1902M"}, {"turn": "F1902M"}],
                ),
                sample(
                    "S1902M",
                    "AUSTRIA",
                    full_public_chat,
                    {"AUSTRIA": {"messaged": ["TURKEY"], "ignored": [], "message_count": 1}},
                    {"AUSTRIA": 2},
                    history=[{"turn": "S1901M"}, {"turn": "S1902M"}],
                ),
            ]
        }

        repaired, stats = repair_game_data(data, public_chat_window=2)

        first_game_history = repaired["training_data"][1]["observation"]["history"]
        self.assertEqual(first_game_history[0]["communications"]["AUSTRIA"]["messaged"], ["ITALY"])
        self.assertEqual(first_game_history[0]["communication_shift"]["AUSTRIA"], 1)
        self.assertEqual(first_game_history[1]["communications"]["AUSTRIA"]["messaged"], ["GERMANY"])

        second_game_history = repaired["training_data"][2]["observation"]["history"]
        self.assertEqual(second_game_history[1]["communications"]["AUSTRIA"]["messaged"], ["TURKEY"])

        self.assertEqual(stats["history_samples_backfilled"], 3)
        self.assertEqual(stats["history_turns_backfilled"], 4)

    def test_assigns_game_ids_and_game_step_indices(self):
        data = {
            "training_data": [
                sample("S1902M", "AUSTRIA", [], {"AUSTRIA": {"messaged": [], "ignored": [], "message_count": 0}}),
                sample("F1902M", "ENGLAND", [], {"ENGLAND": {"messaged": [], "ignored": [], "message_count": 0}}),
                sample("S1901M", "FRANCE", [], {"FRANCE": {"messaged": [], "ignored": [], "message_count": 0}}),
            ]
        }

        repaired, stats = repair_game_data(data, public_chat_window=5)

        observations = [sample["observation"] for sample in repaired["training_data"]]
        self.assertEqual(
            [(obs["game_id"], obs["game_step_index"]) for obs in observations],
            [(0, 0), (0, 1), (1, 0)],
        )
        self.assertEqual(stats["game_samples_annotated"], 3)


if __name__ == "__main__":
    unittest.main()
