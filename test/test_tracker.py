import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from human_detector.tracker import Detection, IoUTracker, TargetSelector, iou  # noqa: E402


def det(x1, y1, x2, y2, score=0.9):
    return Detection((x1, y1, x2, y2), score)


class IoUTest(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint(self):
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_touching_edges(self):
        self.assertEqual(iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)

    def test_half_overlap(self):
        # 50 overlap / (100 + 100 - 50) = 1/3
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1.0 / 3.0)


class TrackerTest(unittest.TestCase):
    def test_confirmed_after_min_hits(self):
        tr = IoUTracker(min_hits=3)
        self.assertEqual(tr.update([det(0, 0, 50, 100)]), [])
        self.assertEqual(tr.update([det(2, 0, 52, 100)]), [])
        confirmed = tr.update([det(4, 0, 54, 100)])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].id, 1)

    def test_id_persists_while_moving(self):
        tr = IoUTracker(min_hits=1, smoothing=1.0)
        ids = set()
        for step in range(20):
            x = step * 5
            ids.update(t.id for t in tr.update([det(x, 0, x + 50, 100)]))
        self.assertEqual(ids, {1})

    def test_two_people_keep_separate_ids(self):
        tr = IoUTracker(min_hits=1, smoothing=1.0)
        for step in range(10):
            a = det(step * 3, 0, step * 3 + 40, 100)
            b = det(300 - step * 3, 0, 340 - step * 3, 100)
            tracks = tr.update([b, a] if step % 2 else [a, b])  # order must not matter
        by_x = sorted(tracks, key=lambda t: t.box[0])
        self.assertEqual([t.id for t in by_x], [1, 2])

    def test_short_occlusion_keeps_id(self):
        tr = IoUTracker(min_hits=1, max_misses=5)
        tr.update([det(0, 0, 50, 100)])
        for _ in range(4):
            self.assertEqual(tr.update([]), [])  # missing, not reported, but kept
        self.assertEqual([t.id for t in tr.update([det(1, 0, 51, 100)])], [1])

    def test_long_occlusion_drops_track(self):
        tr = IoUTracker(min_hits=1, max_misses=3)
        tr.update([det(0, 0, 50, 100)])
        for _ in range(4):
            tr.update([])
        self.assertEqual(tr.tracks, [])
        self.assertEqual([t.id for t in tr.update([det(0, 0, 50, 100)])], [2])

    def test_false_positive_never_confirmed(self):
        tr = IoUTracker(min_hits=3)
        self.assertEqual(tr.update([det(0, 0, 50, 100)]), [])
        self.assertEqual(tr.update([det(400, 0, 450, 100)]), [])
        self.assertEqual(tr.update([]), [])

    def test_smoothing(self):
        tr = IoUTracker(min_hits=1, smoothing=0.5)
        tr.update([det(0, 0, 100, 100)])
        t = tr.update([det(10, 0, 110, 100)])[0]
        self.assertAlmostEqual(t.box[0], 5.0)

    def test_bad_smoothing(self):
        with self.assertRaises(ValueError):
            IoUTracker(smoothing=0.0)


class TargetSelectorTest(unittest.TestCase):
    def setUp(self):
        self.tr = IoUTracker(min_hits=1, max_misses=3, smoothing=1.0)
        self.sel = TargetSelector()

    def step(self, *dets):
        confirmed = self.tr.update(list(dets))
        return self.sel.select(self.tr, confirmed)

    def test_picks_largest(self):
        target = self.step(det(0, 0, 20, 40), det(100, 0, 200, 200))
        self.assertEqual(target.id, 2)

    def test_sticks_with_target_when_someone_bigger_appears(self):
        self.step(det(0, 0, 50, 100))
        target = self.step(det(0, 0, 50, 100), det(200, 0, 400, 300))
        self.assertEqual(target.id, 1)

    def test_none_during_occlusion_then_resumes(self):
        self.step(det(0, 0, 50, 100), det(200, 0, 230, 60))
        self.assertIsNone(self.step(det(200, 0, 230, 60)))  # target hidden, don't switch
        self.assertEqual(self.step(det(0, 0, 50, 100), det(200, 0, 230, 60)).id, 1)

    def test_switches_after_target_dropped(self):
        self.step(det(0, 0, 50, 100), det(200, 0, 230, 60))
        for _ in range(4):
            target = self.step(det(200, 0, 230, 60))
        self.assertEqual(target.id, 2)

    def test_manual_lock(self):
        self.step(det(0, 0, 50, 100), det(200, 0, 230, 60))
        self.sel.lock(2)
        self.assertEqual(self.step(det(0, 0, 50, 100), det(200, 0, 230, 60)).id, 2)
        self.sel.lock(None)
        self.assertEqual(self.step(det(0, 0, 50, 100), det(200, 0, 230, 60)).id, 2)

    def test_locked_target_gone_waits(self):
        self.step(det(0, 0, 50, 100), det(200, 0, 230, 60))
        self.sel.lock(2)
        for _ in range(5):
            target = self.step(det(0, 0, 50, 100))
        self.assertIsNone(target)

    def test_nothing_seen(self):
        self.assertIsNone(self.step())


if __name__ == "__main__":
    unittest.main()
