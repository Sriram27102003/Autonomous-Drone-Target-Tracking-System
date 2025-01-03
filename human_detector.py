#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from std_msgs.msg import String
import tf
from tf.transformations import euler_from_quaternion
import random

class TurtleBot3HumanDetectorSLAM:
    def __init__(self):
        self.bridge = CvBridge()
        # Update the camera topic to match TurtleBot3's simulated camera
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.alert_pub = rospy.Publisher("/human_detection_alert", String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.teleop_sub = rospy.Subscriber("/cmd_vel_teleop", Twist, self.teleop_callback)

        # Initialize HOG descriptor
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Robot state
        self.position = [0, 0, 0]
        self.orientation = 0
        self.laser_ranges = []

        # Control mode
        self.autonomous_mode = True

        # Debug flag
        self.debug = rospy.get_param('~debug', False)

        rospy.loginfo("TurtleBot3 Human Detector SLAM node initialized")

    def image_callback(self, data):
        if self.debug:
            rospy.loginfo("Received image data")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Detect humans in the image
        humans, _ = self.hog.detectMultiScale(cv_image, winStride=(8, 8), padding=(32, 32), scale=1.05)

        # Draw bounding boxes around detected humans
        for (x, y, w, h) in humans:
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # If humans are detected, publish an alert
        if len(humans) > 0:
            alert_msg = f"Human detected! Count: {len(humans)}"
            self.alert_pub.publish(alert_msg)
            rospy.loginfo(alert_msg)

        # Display the result
        cv2.imshow("TurtleBot3 Human Detection", cv_image)
        cv2.waitKey(3)

    # ... (rest of the methods remain the same)

def main():
    rospy.init_node('turtlebot3_human_detector_slam', anonymous=True)
    detector = TurtleBot3HumanDetectorSLAM()
    try:
        detector.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down turtlebot3_human_detector_slam node")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()