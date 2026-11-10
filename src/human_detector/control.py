"""Geometry and velocity control for following a person and exploring.

Angles follow ROS conventions: radians, positive = counter-clockwise (left).
"""

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def pixel_to_bearing(u, cx, fx):
    """Bearing (rad) of image column ``u`` for a pinhole camera.

    ``cx`` is the principal point and ``fx`` the focal length in pixels
    (``K[2]`` and ``K[0]`` of sensor_msgs/CameraInfo). Pixels left of centre
    give a positive (leftward) bearing.
    """
    return math.atan2(cx - u, fx)


def distance_from_box_height(box_height_px, fy, person_height_m=1.7):
    """Rough range to a standing person from their bounding-box height."""
    if box_height_px <= 1.0 or fy <= 0.0:
        return None
    return person_height_m * fy / box_height_px


def _valid(r, range_min, range_max):
    return r is not None and not math.isnan(r) and not math.isinf(r) and range_min <= r <= range_max


def scan_sector(ranges, angle_min, angle_increment, center, half_width,
                range_min=0.0, range_max=float("inf")):
    """Valid readings of a LaserScan within ``center +/- half_width`` radians.

    Handles scans that wrap around (e.g. TurtleBot3's 0..2*pi scan).
    """
    n = len(ranges)
    if n == 0 or angle_increment <= 0.0:
        return []
    span = int(math.ceil(half_width / angle_increment))
    mid = int(round((center - angle_min) / angle_increment))
    full_circle = abs(n * angle_increment - 2.0 * math.pi) < 2.0 * angle_increment
    out = []
    for i in range(mid - span, mid + span + 1):
        if full_circle:
            i %= n
        elif i < 0 or i >= n:
            continue
        r = ranges[i]
        if _valid(r, range_min, range_max):
            out.append(r)
    return out


def median(values):
    if not values:
        return None
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def fuse_distance(lidar_m, camera_m, tolerance=0.5):
    """Combine lidar range and camera estimate.

    The lidar is precise but may hit something behind or beside the person
    (or not see them at all, e.g. Gazebo actors have no collision). It is
    trusted only when it agrees with the camera estimate within ``tolerance``
    (as a fraction of the camera estimate).
    """
    if camera_m is None:
        return lidar_m
    if lidar_m is None:
        return camera_m
    if abs(lidar_m - camera_m) <= tolerance * camera_m:
        return lidar_m
    return camera_m


class FollowController(object):
    """Proportional controller that keeps a person centred at a set distance."""

    def __init__(self, target_distance=1.2, kp_linear=0.5, kp_angular=1.8,
                 max_linear=0.22, max_angular=1.5, distance_deadband=0.1,
                 angular_deadband=0.03, safety_distance=0.35, allow_reverse=True,
                 search_angular=0.5):
        self.target_distance = target_distance
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.distance_deadband = distance_deadband
        self.angular_deadband = angular_deadband
        self.safety_distance = safety_distance
        self.allow_reverse = allow_reverse
        self.search_angular = search_angular
        self.last_bearing = 0.0

    def follow(self, bearing, distance, front_clearance=None):
        """Return ``(linear, angular)`` velocity to follow the target.

        ``distance`` may be ``None`` (rotate only). ``front_clearance`` is the
        closest lidar reading straight ahead; forward motion stops when it is
        below ``safety_distance``.
        """
        self.last_bearing = bearing

        angular = 0.0
        if abs(bearing) > self.angular_deadband:
            angular = clamp(self.kp_angular * bearing, -self.max_angular, self.max_angular)

        linear = 0.0
        if distance is not None:
            error = distance - self.target_distance
            if abs(error) > self.distance_deadband:
                linear = self.kp_linear * error
            # Turn first, drive second: slow down when the person is off to the side.
            linear *= max(0.0, math.cos(bearing)) ** 2
            low = -self.max_linear if self.allow_reverse else 0.0
            linear = clamp(linear, low, self.max_linear)

        if linear > 0.0 and front_clearance is not None and front_clearance < self.safety_distance:
            linear = 0.0
        return linear, angular

    def search(self):
        """Rotate towards where the target was last seen."""
        direction = 1.0 if self.last_bearing >= 0.0 else -1.0
        return 0.0, direction * self.search_angular


class Explorer(object):
    """Simple reactive wandering for building a map: drive, veer away from obstacles."""

    def __init__(self, forward_speed=0.15, turn_speed=0.8, obstacle_distance=0.5,
                 slow_distance=1.0):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.obstacle_distance = obstacle_distance
        self.slow_distance = slow_distance
        self._turn_dir = 0.0  # keep turning the same way until clear, avoids dithering

    def command(self, front, left, right):
        """``front``/``left``/``right``: closest range in each sector (None = clear)."""
        inf = float("inf")
        front = inf if front is None else front
        left = inf if left is None else left
        right = inf if right is None else right

        if front < self.obstacle_distance:
            if self._turn_dir == 0.0:
                self._turn_dir = 1.0 if left >= right else -1.0
            return 0.0, self._turn_dir * self.turn_speed

        self._turn_dir = 0.0
        speed = self.forward_speed
        if front < self.slow_distance:
            span = self.slow_distance - self.obstacle_distance
            speed *= clamp((front - self.obstacle_distance) / span, 0.3, 1.0)
        # Gentle drift away from the nearer side wall.
        steer = 0.0
        if min(left, right) < self.slow_distance:
            steer = 0.4 * self.turn_speed * (1.0 if left > right else -1.0)
        return speed, steer
