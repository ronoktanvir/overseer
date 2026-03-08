import unittest

from judge import StrategyJudge, parse_binary_score, parse_similarity_score


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [type("TextBlock", (), {"text": text})()]


class _FakeMessages:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def create(self, **kwargs):
        return _FakeResponse(self.outputs.pop(0))


class _FakeClient:
    def __init__(self, outputs):
        self.messages = _FakeMessages(outputs)


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_binary_score(self):
        self.assertEqual(parse_binary_score("0"), 0)
        self.assertEqual(parse_binary_score("1"), 1)

    def test_parse_binary_score_rejects_non_binary_output(self):
        with self.assertRaises(ValueError):
            parse_binary_score("0.5")

    def test_parse_similarity_score(self):
        self.assertEqual(parse_similarity_score("0"), 0)
        self.assertEqual(parse_similarity_score("73"), 73)
        self.assertEqual(parse_similarity_score("100"), 100)

    def test_parse_similarity_score_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_similarity_score("101")

    async def test_score_prediction(self):
        judge = StrategyJudge(client=_FakeClient(["1"]))
        score = await judge.score_prediction(
            target_player="FRANCE",
            true_strategy="Expand north while keeping England calm.",
            predicted_strategy="France wants to move north and avoid conflict with England.",
        )
        self.assertEqual(score, 1)

    async def test_score_similarity(self):
        judge = StrategyJudge(client=_FakeClient(["83"]))
        score = await judge.score_similarity(
            target_player="FRANCE",
            true_strategy="Expand north while keeping England calm.",
            predicted_strategy="France wants to move north and avoid conflict with England.",
        )
        self.assertEqual(score, 83)

    async def test_score_turn_returns_average_reward(self):
        judge = StrategyJudge(client=_FakeClient(["1", "0", "1"]))
        result = await judge.score_turn(
            predictions={
                "FRANCE": "Push north carefully.",
                "GERMANY": "Attack west immediately.",
                "ITALY": "Stay flexible and defend.",
            },
            true_strategies={
                "FRANCE": "Push north carefully.",
                "GERMANY": "Defend in the west.",
                "ITALY": "Stay flexible and defend.",
            },
        )
        self.assertEqual(result["scores"], {"FRANCE": 1.0, "GERMANY": 0.0, "ITALY": 1.0})
        self.assertAlmostEqual(result["average_reward"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
