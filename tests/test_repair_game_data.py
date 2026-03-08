import unittest

from repair_game_data import repair_game_data


def sample(turn, target, public_chat, communications):
    return {
        "observation": {
            "target_player": target,
            "turn": turn,
            "current_state": {"orders": {}, "units": {}, "supply_centers": {}, "conflicts": {}},
            "history": [],
            "communications": communications,
            "communication_shift": {},
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


if __name__ == "__main__":
    unittest.main()
