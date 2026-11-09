import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from human_detector.control import (Explorer, FollowController,  # noqa: E402
                                    distance_from_box_height, fuse_distance, median,
                                    pixel_to_bearing, scan_sector)

INF = float("inf")
NAN = float("nan")


class GeometryTest(unittest.TestCase):
    def test_bearing_center_is_zero(self):
        self.assertAlmostEqual(pixel_to_bearing(320, 320, 500), 0.0)

    def test_bearing_left_is_positive(self):
        self.assertGreater(pixel_to_bearing(100, 320, 500), 0.0)
        self.assertLess(pixel_to_bearing(600, 320, 500), 0.0)

    def test_bearing_value(self):
        self.assertAlmostEqual(pixel_to_bearing(0, 500, 500), math.pi / 4)

    def test_distance_from_box(self):
        self.assertAlmostEqual(distance_from_box_height(170, 500, 1.7), 5.0)
        self.assertIsNone(distance_from_box_height(0, 500))

    def test_median(self):
        self.assertIsNone(median([]))
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 3, 2]), 2.5)


class ScanSectorTest(unittest.TestCase):
    def full_scan(self, n=360):
        # TurtleBot3 style: 0..2pi, index i = i degrees
        return [float(i) for i in range(n)], 0.0, 2 * math.pi / n

    def test_wraps_around_zero(self):
        ranges, amin, inc = self.full_scan()
        vals = scan_sector(ranges, amin, inc, 0.0, math.radians(2), 0.0, 1000.0)
        self.assertEqual(sorted(vals), [0.0, 1.0, 2.0, 358.0, 359.0])

    def test_negative_bearing(self):
        ranges, amin, inc = self.full_scan()
        vals = scan_sector(ranges, amin, inc, math.radians(-90), math.radians(1), 0.0, 1000.0)
        self.assertEqual(sorted(vals), [269.0, 270.0, 271.0])

    def test_filters_invalid(self):
        ranges = [1.0, INF, NAN, 0.01, 50.0]
        vals = scan_sector(ranges, -0.02, 0.01, 0.0, 0.03, 0.12, 3.5)
        self.assertEqual(vals, [1.0])

    def test_partial_scan_does_not_wrap(self):
        # -90..+90 degrees, 181 samples
        ranges = [float(i) for i in range(181)]
        vals = scan_sector(ranges, -math.pi / 2, math.radians(1), math.radians(-90),
                           math.radians(2), 0.0, 1000.0)
        self.assertEqual(sorted(vals), [0.0, 1.0, 2.0])

    def test_empty(self):
        self.assertEqual(scan_sector([], 0.0, 0.01, 0.0, 0.1), [])


class FuseDistanceTest(unittest.TestCase):
    def test_agreeing_lidar_wins(self):
        self.assertEqual(fuse_distance(2.1, 2.5), 2.1)

    def test_lidar_hits_background(self):
        self.assertEqual(fuse_distance(8.0, 2.5), 2.5)

    def test_missing_values(self):
        self.assertEqual(fuse_distance(None, 2.5), 2.5)
        self.assertEqual(fuse_distance(2.0, None), 2.0)
        self.assertIsNone(fuse_distance(None, None))


class FollowControllerTest(unittest.TestCase):
    def setUp(self):
        self.c = FollowController(target_distance=1.0, max_linear=0.22, max_angular=1.5)

    def test_drives_forward_when_far(self):
        v, w = self.c.follow(0.0, 3.0)
        self.assertAlmostEqual(v, 0.22)
        self.assertEqual(w, 0.0)

    def test_backs_off_when_too_close(self):
        v, _ = self.c.follow(0.0, 0.5)
        self.assertLess(v, 0.0)

    def test_no_reverse_option(self):
        c = FollowController(target_distance=1.0, allow_reverse=False)
        self.assertEqual(c.follow(0.0, 0.5)[0], 0.0)

    def test_holds_inside_deadband(self):
        self.assertEqual(self.c.follow(0.0, 1.05), (0.0, 0.0))

    def test_turns_towards_target(self):
        self.assertGreater(self.c.follow(0.3, 1.0)[1], 0.0)
        self.assertLess(self.c.follow(-0.3, 1.0)[1], 0.0)

    def test_angular_clamped(self):
        self.assertEqual(self.c.follow(3.0, 1.0)[1], 1.5)

    def test_slows_when_target_to_the_side(self):
        straight, _ = self.c.follow(0.0, 1.3)
        side, _ = self.c.follow(0.6, 1.3)
        self.assertLess(side, straight)
        self.assertAlmostEqual(self.c.follow(math.pi / 2, 5.0)[0], 0.0)

    def test_safety_stop(self):
        v, w = self.c.follow(0.2, 3.0, front_clearance=0.2)
        self.assertEqual(v, 0.0)
        self.assertGreater(w, 0.0)  # can still turn

    def test_safety_allows_reverse(self):
        self.assertLess(self.c.follow(0.0, 0.4, front_clearance=0.2)[0], 0.0)

    def test_unknown_distance_rotates_only(self):
        v, w = self.c.follow(0.4, None)
        self.assertEqual(v, 0.0)
        self.assertGreater(w, 0.0)

    def test_search_turns_towards_last_seen(self):
        self.c.follow(-0.5, 2.0)
        self.assertLess(self.c.search()[1], 0.0)
        self.c.follow(0.5, 2.0)
        self.assertGreater(self.c.search()[1], 0.0)


class ExplorerTest(unittest.TestCase):
    def setUp(self):
        self.e = Explorer(forward_speed=0.15, turn_speed=0.8, obstacle_distance=0.5,
                          slow_distance=1.0)

    def test_open_space_full_speed(self):
        self.assertEqual(self.e.command(None, None, None), (0.15, 0.0))

    def test_obstacle_turns_to_open_side(self):
        v, w = self.e.command(0.3, 2.0, 0.6)
        self.assertEqual(v, 0.0)
        self.assertGreater(w, 0.0)  # left is more open

    def test_keeps_turn_direction_until_clear(self):
        self.e.command(0.3, 2.0, 0.6)
        _, w = self.e.command(0.3, 0.6, 2.0)  # sides flipped mid-turn
        self.assertGreater(w, 0.0)
        self.e.command(3.0, None, None)
        _, w = self.e.command(0.3, 0.6, 2.0)
        self.assertLess(w, 0.0)

    def test_slows_near_obstacle(self):
        v, _ = self.e.command(0.75, None, None)
        self.assertLess(v, 0.15)
        self.assertGreater(v, 0.0)

    def test_veers_from_side_wall(self):
        _, w = self.e.command(None, 0.6, None)
        self.assertLess(w, 0.0)  # wall on the left -> turn right


if __name__ == "__main__":
    unittest.main()
