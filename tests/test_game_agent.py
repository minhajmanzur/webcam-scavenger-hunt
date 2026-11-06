import time
import unittest

import numpy as np

from game_agent import Detection, HuntAgent, HuntTarget


class HuntAgentTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((50, 50, 3), dtype=np.uint8)
        self.detection = Detection("book", 0.8, (0.0, 0.0, 50.0, 50.0))

    def test_stable_target_scores_and_advances(self):
        agent = HuntAgent([HuntTarget("book"), HuntTarget("cup")])
        now = time.perf_counter()
        agent.reset(now=now)

        agent.observe(
            [self.detection],
            {"book": (True, 1.0)},
            self.frame,
            now=now + 1,
        )

        self.assertEqual(agent.score, 100)
        self.assertEqual(agent.found, ["book"])
        self.assertEqual(agent.current_target, HuntTarget("cup"))

    def test_stuck_player_gets_a_hint(self):
        agent = HuntAgent([HuntTarget("book")])
        now = time.perf_counter()
        agent.reset(now=now)

        agent.observe([], {"book": (False, 0.0)}, self.frame, now=now + 13)

        self.assertEqual(agent.hints_given, 1)
        self.assertIn("book", agent.last_hint)

    def test_color_tool_accepts_matching_hue(self):
        agent = HuntAgent([HuntTarget("book")])
        blue_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        blue_frame[:, :] = (255, 0, 0)  # BGR blue

        matched, fraction = agent.check_color(blue_frame, self.detection, "blue")

        self.assertTrue(matched)
        self.assertEqual(fraction, 1.0)

    def test_color_target_is_not_awarded_for_wrong_color(self):
        agent = HuntAgent([HuntTarget("book", color="blue")])
        now = time.perf_counter()
        agent.reset(now=now)
        red_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        red_frame[:, :] = (0, 0, 255)  # BGR red

        agent.observe(
            [self.detection],
            {"book": (True, 1.0)},
            red_frame,
            now=now + 1,
        )

        self.assertEqual(agent.score, 0)
        self.assertEqual(agent.current_target, HuntTarget("book", color="blue"))


if __name__ == "__main__":
    unittest.main()
