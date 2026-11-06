import unittest

from presence import PresenceStabilizer


class PresenceStabilizerTests(unittest.TestCase):
    def test_confirms_after_one_second(self):
        tracker = PresenceStabilizer(("book",), persistence_seconds=1.0)

        for frame_index in range(11):
            confirmed, progress = tracker.update(
                {"book"}, now=10.0 + frame_index / 10
            )["book"]

        self.assertTrue(confirmed)
        self.assertEqual(progress, 1.0)

    def test_tolerates_a_brief_detection_flicker(self):
        tracker = PresenceStabilizer(("book",), persistence_seconds=1.0)
        tracker.update({"book"}, now=10.0)
        tracker.update(set(), now=10.2)

        confirmed, progress = tracker.update({"book"}, now=10.3)["book"]

        self.assertFalse(confirmed)
        self.assertAlmostEqual(progress, 0.3)
        for frame_index in range(1, 8):
            confirmed, _ = tracker.update(
                {"book"}, now=10.3 + frame_index / 10
            )["book"]
        self.assertTrue(confirmed)

    def test_resets_after_long_detection_gap(self):
        tracker = PresenceStabilizer(("book",), persistence_seconds=1.0)
        tracker.update({"book"}, now=10.0)

        self.assertEqual(tracker.update(set(), now=10.36)["book"], (False, 0.0))
        self.assertEqual(tracker.update({"book"}, now=10.4)["book"], (False, 0.0))


if __name__ == "__main__":
    unittest.main()
