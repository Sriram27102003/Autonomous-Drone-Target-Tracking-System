#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

# YOLOv8 model loading (ensure that ultralytics is installed)
model = YOLO('yolov8n.pt')  # You can change this to yolov8s.pt, yolov8m.pt, etc., depending on the model size you prefer

# Initialize CvBridge
bridge = CvBridge()

def image_callback(ros_image):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridgeError: {e}")
        return

    # Perform YOLOv8 detection
    results = model(cv_image)

    # Draw the bounding boxes and labels on the image
    detection_image = results[0].plot()

    # Resize the images to 512x512
    original_resized = cv2.resize(cv_image, (512, 512))
    detection_resized = cv2.resize(detection_image, (512, 512))

    # Display the original image and detection image
    cv2.imshow('Original Image', original_resized)
    cv2.imshow('Detection Image', detection_resized)
    
    # Press 'q' to quit the OpenCV windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User exited")

def main():
    # Initialize ROS node
    rospy.init_node('yolov8_object_detection', anonymous=True)

    # Subscribe to the /camera/rgb/image_raw topic
    rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)

    try:
        # Spin to keep the script running
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
