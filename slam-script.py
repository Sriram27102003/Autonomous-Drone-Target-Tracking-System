#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class SlamNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('my_slam_node', anonymous=True)
        
        # Create a publisher to send velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Create a subscriber for the camera feed
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        
        # Create a CvBridge object
        self.bridge = CvBridge()
        
        # Create an OpenCV window
        cv2.namedWindow("Robot POV", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot POV", 512, 512)
        
        # Store the latest image
        self.latest_image = None
        
        # Rate at which to send commands
        self.rate = rospy.Rate(10)

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Resize the image to 512x512
            self.latest_image = cv2.resize(cv_image, (512, 512))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def run(self):
        while not rospy.is_shutdown():
            # Create a Twist message for moving forward
            move_cmd = Twist()
            move_cmd.linear.x = 0.1  # Move forward at 0.1 m/s
            move_cmd.angular.z = 0.0  # No rotation
            
            # Publish the command
            self.cmd_vel_pub.publish(move_cmd)
            
            # Display the latest image if available
            if self.latest_image is not None:
                cv2.imshow("Robot POV", self.latest_image)
                cv2.waitKey(1)
            
            # Sleep to maintain the loop rate
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = SlamNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()