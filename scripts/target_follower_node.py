#!/usr/bin/env python3
"""Drive the robot to follow the tracked person.

Subscribes:
  target (vision_msgs/Detection2D)     from person_tracker (~target)
  camera_info (sensor_msgs/CameraInfo) intrinsics, used to turn pixels into angles
  scan (sensor_msgs/LaserScan)         range to the person and obstacle safety stop

Publishes:
  cmd_vel (geometry_msgs/Twist)

Services:
  ~enable (std_srvs/SetBool)           start/stop following (robot stops when disabled)
"""

import math
import threading

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, LaserScan
from std_srvs.srv import SetBool, SetBoolResponse
from vision_msgs.msg import Detection2D

from human_detector.control import (FollowController, distance_from_box_height, fuse_distance,
                                    median, pixel_to_bearing, scan_sector)


class TargetFollowerNode(object):
    def __init__(self):
        self.controller = FollowController(
            target_distance=rospy.get_param("~target_distance", 1.2),
            kp_linear=rospy.get_param("~kp_linear", 0.5),
            kp_angular=rospy.get_param("~kp_angular", 1.8),
            max_linear=rospy.get_param("~max_linear", 0.22),
            max_angular=rospy.get_param("~max_angular", 1.5),
            distance_deadband=rospy.get_param("~distance_deadband", 0.1),
            angular_deadband=rospy.get_param("~angular_deadband", 0.03),
            safety_distance=rospy.get_param("~safety_distance", 0.35),
            allow_reverse=rospy.get_param("~allow_reverse", True),
            search_angular=rospy.get_param("~search_angular", 0.5))
        self.enabled = rospy.get_param("~enabled", True)
        self.search_when_lost = rospy.get_param("~search_when_lost", True)
        self.search_timeout = rospy.Duration(rospy.get_param("~search_timeout", 20.0))
        self.target_timeout = rospy.Duration(rospy.get_param("~target_timeout", 0.7))
        self.person_height = rospy.get_param("~person_height", 1.7)
        self.scan_window = math.radians(rospy.get_param("~scan_window_deg", 4.0))
        self.front_half_width = math.radians(rospy.get_param("~front_half_width_deg", 20.0))
        self.distance_tolerance = rospy.get_param("~distance_tolerance", 0.5)

        self.lock = threading.Lock()
        self.target = None
        self.target_time = rospy.Time(0)
        self.camera = None   # (fx, fy, cx)
        self.scan = None
        self.was_following = False

        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("target", Detection2D, self.on_target, queue_size=1)
        rospy.Subscriber("camera_info", CameraInfo, self.on_camera_info, queue_size=1)
        rospy.Subscriber("scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Service("~enable", SetBool, self.on_enable)

        rate = rospy.get_param("~rate", 10.0)
        rospy.Timer(rospy.Duration(1.0 / rate), self.step)
        rospy.on_shutdown(self.stop)
        rospy.loginfo("target_follower: ready (enabled=%s, target distance %.2f m)",
                      self.enabled, self.controller.target_distance)

    def on_target(self, msg):
        with self.lock:
            self.target = msg
            self.target_time = rospy.Time.now()

    def on_camera_info(self, msg):
        if msg.K[0] > 0.0:
            self.camera = (msg.K[0], msg.K[4], msg.K[2])

    def on_scan(self, msg):
        self.scan = msg

    def on_enable(self, req):
        self.enabled = req.data
        if not self.enabled:
            self.stop()
        return SetBoolResponse(success=True, message="following %s" % ("on" if req.data else "off"))

    def stop(self):
        self.cmd_pub.publish(Twist())

    def sector(self, center, half_width):
        s = self.scan
        if s is None:
            return []
        return scan_sector(s.ranges, s.angle_min, s.angle_increment, center, half_width,
                           s.range_min, s.range_max)

    def step(self, _event):
        if not self.enabled:
            return
        if self.camera is None:
            rospy.logwarn_throttle(5, "target_follower: waiting for camera_info")
            return

        with self.lock:
            target, stamp = self.target, self.target_time
        now = rospy.Time.now()
        fresh = target is not None and now - stamp < self.target_timeout

        if fresh:
            fx, fy, cx = self.camera
            bearing = pixel_to_bearing(target.bbox.center.x, cx, fx)
            camera_range = distance_from_box_height(target.bbox.size_y, fy, self.person_height)
            lidar_range = median(self.sector(bearing, self.scan_window))
            distance = fuse_distance(lidar_range, camera_range, self.distance_tolerance)
            front = self.sector(0.0, self.front_half_width)
            linear, angular = self.controller.follow(bearing, distance, min(front) if front else None)
            if not self.was_following:
                rospy.loginfo("target_follower: following track %s",
                              target.results[0].id if target.results else "?")
            self.was_following = True
            rospy.logdebug("bearing %.2f rad, distance %s m -> v %.2f, w %.2f",
                           bearing, distance, linear, angular)
        else:
            if self.was_following:
                rospy.loginfo("target_follower: target lost%s",
                              ", searching" if self.search_when_lost else "")
            self.was_following = False
            searching = (self.search_when_lost and target is not None
                         and now - stamp < self.target_timeout + self.search_timeout)
            linear, angular = self.controller.search() if searching else (0.0, 0.0)

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main():
    rospy.init_node("target_follower")
    TargetFollowerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
