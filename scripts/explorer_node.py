#!/usr/bin/env python3
"""Wander around while avoiding obstacles, so SLAM can build a map hands-free.

Subscribes: scan (sensor_msgs/LaserScan)
Publishes:  cmd_vel (geometry_msgs/Twist)
Services:   ~enable (std_srvs/SetBool)
"""

import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool, SetBoolResponse

from human_detector.control import Explorer, scan_sector


class ExplorerNode(object):
    def __init__(self):
        self.explorer = Explorer(
            forward_speed=rospy.get_param("~forward_speed", 0.15),
            turn_speed=rospy.get_param("~turn_speed", 0.8),
            obstacle_distance=rospy.get_param("~obstacle_distance", 0.5),
            slow_distance=rospy.get_param("~slow_distance", 1.0))
        self.front = math.radians(rospy.get_param("~front_half_width_deg", 30.0))
        self.enabled = rospy.get_param("~enabled", True)
        self.scan = None

        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Service("~enable", SetBool, self.on_enable)
        rospy.Timer(rospy.Duration(0.1), self.step)
        rospy.on_shutdown(lambda: self.cmd_pub.publish(Twist()))

    def on_scan(self, msg):
        self.scan = msg

    def on_enable(self, req):
        self.enabled = req.data
        if not self.enabled:
            self.cmd_pub.publish(Twist())
        return SetBoolResponse(success=True, message="exploring %s" % ("on" if req.data else "off"))

    def closest(self, center, half_width):
        s = self.scan
        values = scan_sector(s.ranges, s.angle_min, s.angle_increment, center, half_width,
                             s.range_min, s.range_max)
        return min(values) if values else None

    def step(self, _event):
        if not self.enabled or self.scan is None:
            return
        front = self.closest(0.0, self.front)
        left = self.closest(math.radians(70), math.radians(35))
        right = self.closest(math.radians(-70), math.radians(35))
        linear, angular = self.explorer.command(front, left, right)
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main():
    rospy.init_node("explorer")
    ExplorerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
