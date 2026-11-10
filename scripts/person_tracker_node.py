#!/usr/bin/env python3
"""Detect and track people in the robot's camera feed.

Subscribes:
  image (sensor_msgs/Image)              camera feed, remap to e.g. /camera/rgb/image_raw
  ~select_target (std_msgs/Int32)        track ID to follow, or -1 for automatic choice

Publishes:
  ~detections (vision_msgs/Detection2DArray)  all confirmed people; results[0].id = track ID
  ~target (vision_msgs/Detection2D)           the person being followed (only when visible)
  ~annotated (sensor_msgs/Image)              camera image with boxes and IDs drawn
  /human_detection_alert (std_msgs/String)    human-readable message when the count changes
"""

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

from human_detector.detectors import make_detector
from human_detector.tracker import IoUTracker, TargetSelector

TARGET_COLOR = (0, 0, 255)
OTHER_COLOR = (0, 200, 0)


class PersonTrackerNode(object):
    def __init__(self):
        backend = rospy.get_param("~backend", "yolo")
        if backend == "yolo":
            detector_args = dict(
                model=rospy.get_param("~model", "yolov8n.pt"),
                confidence=rospy.get_param("~confidence", 0.4),
                device=rospy.get_param("~device", "") or None,
                image_size=rospy.get_param("~image_size", 640))
        else:
            detector_args = dict(confidence=rospy.get_param("~confidence", 0.3))
        rospy.loginfo("person_tracker: loading %s detector...", backend)
        self.detector = make_detector(backend, **detector_args)

        self.tracker = IoUTracker(
            iou_threshold=rospy.get_param("~iou_threshold", 0.3),
            max_misses=rospy.get_param("~max_misses", 15),
            min_hits=rospy.get_param("~min_hits", 3),
            smoothing=rospy.get_param("~smoothing", 0.6))
        self.selector = TargetSelector()

        max_rate = rospy.get_param("~max_rate", 10.0)
        self.min_period = rospy.Duration(1.0 / max_rate) if max_rate > 0 else rospy.Duration(0)
        self.show_window = rospy.get_param("~show_window", False)
        self.last_processed = rospy.Time(0)
        self.last_count = -1
        self.bridge = CvBridge()

        self.detections_pub = rospy.Publisher("~detections", Detection2DArray, queue_size=1)
        self.target_pub = rospy.Publisher("~target", Detection2D, queue_size=1)
        self.annotated_pub = rospy.Publisher("~annotated", Image, queue_size=1)
        self.alert_pub = rospy.Publisher("/human_detection_alert", String, queue_size=10)

        rospy.Subscriber("~select_target", Int32, self.on_select_target)
        # queue_size=1 + large buffer: always process the newest frame, never a backlog.
        rospy.Subscriber("image", Image, self.on_image, queue_size=1, buff_size=2 ** 24)
        rospy.loginfo("person_tracker: ready, waiting for images on %s",
                      rospy.resolve_name("image"))

    def on_select_target(self, msg):
        if msg.data < 0:
            self.selector.lock(None)
            rospy.loginfo("person_tracker: automatic target selection")
        else:
            self.selector.lock(msg.data)
            rospy.loginfo("person_tracker: locked on track %d", msg.data)

    def on_image(self, msg):
        elapsed = msg.header.stamp - self.last_processed
        if rospy.Duration(0) <= elapsed < self.min_period:
            return  # rate limit (a negative gap means the sim clock was reset)
        self.last_processed = msg.header.stamp

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr_throttle(5, "person_tracker: %s", e)
            return

        detections = self.detector.detect(frame)
        confirmed = self.tracker.update(detections)
        target = self.selector.select(self.tracker, confirmed)

        array = Detection2DArray(header=msg.header)
        array.detections = [self.to_detection_msg(t, msg) for t in confirmed]
        self.detections_pub.publish(array)
        if target is not None:
            self.target_pub.publish(self.to_detection_msg(target, msg))

        if len(confirmed) != self.last_count:
            self.last_count = len(confirmed)
            if confirmed:
                text = "Human detected! Count: %d" % len(confirmed)
                self.alert_pub.publish(text)
                rospy.loginfo("person_tracker: %s", text)

        if self.annotated_pub.get_num_connections() > 0 or self.show_window:
            annotated = self.draw(frame, confirmed, target)
            if self.annotated_pub.get_num_connections() > 0:
                out = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
                out.header = msg.header
                self.annotated_pub.publish(out)
            if self.show_window:
                cv2.imshow("Person tracker", annotated)
                cv2.waitKey(1)

    @staticmethod
    def to_detection_msg(track, image_msg):
        det = Detection2D(header=image_msg.header)
        cx, cy = track.center
        det.bbox = BoundingBox2D()
        det.bbox.center.x = cx
        det.bbox.center.y = cy
        det.bbox.size_x = track.width
        det.bbox.size_y = track.height
        hyp = ObjectHypothesisWithPose()
        hyp.id = track.id
        hyp.score = track.score
        det.results.append(hyp)
        return det

    @staticmethod
    def draw(frame, tracks, target):
        img = frame.copy()
        for t in tracks:
            is_target = target is not None and t.id == target.id
            color = TARGET_COLOR if is_target else OTHER_COLOR
            x1, y1, x2, y2 = (int(v) for v in t.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if is_target else 2)
            label = "%s#%d %.2f" % ("TARGET " if is_target else "", t.id, t.score)
            cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, "people: %d" % len(tracks), (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img


def main():
    rospy.init_node("person_tracker")
    PersonTrackerNode()
    try:
        rospy.spin()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
