"""Person detectors. Each returns a list of ``tracker.Detection``.

* ``YoloPersonDetector`` - YOLOv8 via the ``ultralytics`` package (accurate, needs pip).
* ``HogPersonDetector`` - OpenCV's built-in HOG + SVM people detector (no extra deps, weaker).
"""

from .tracker import Detection

PERSON_CLASS_ID = 0  # "person" in the COCO classes YOLOv8 is trained on


class YoloPersonDetector(object):
    def __init__(self, model="yolov8n.pt", confidence=0.4, device=None, image_size=640):
        from ultralytics import YOLO  # imported here so HOG works without ultralytics
        self.model = YOLO(model)
        self.confidence = confidence
        self.device = device or None
        self.image_size = image_size

    def detect(self, bgr_image):
        results = self.model.predict(
            bgr_image, classes=[PERSON_CLASS_ID], conf=self.confidence,
            device=self.device, imgsz=self.image_size, verbose=False)
        out = []
        if not results:
            return out
        boxes = results[0].boxes
        for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
            out.append(Detection(xyxy, conf))
        return out


class HogPersonDetector(object):
    def __init__(self, confidence=0.3, nms_threshold=0.4, scale=1.05, max_width=640):
        import cv2
        if not hasattr(cv2, "HOGDescriptor"):
            raise ImportError(
                "OpenCV %s has no HOGDescriptor (it left the core module in OpenCV 5). "
                "Use backend:=yolo, or OpenCV 4.x (ROS Noetic ships 4.2)." % cv2.__version__)
        self.cv2 = cv2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.scale = scale
        self.max_width = max_width

    def detect(self, bgr_image):
        cv2 = self.cv2
        h, w = bgr_image.shape[:2]
        ratio = 1.0
        if w > self.max_width:  # HOG is slow; work on a smaller copy
            ratio = self.max_width / float(w)
            bgr_image = cv2.resize(bgr_image, (int(w * ratio), int(h * ratio)))

        rects, weights = self.hog.detectMultiScale(
            bgr_image, winStride=(8, 8), padding=(16, 16), scale=self.scale)
        if len(rects) == 0:
            return []
        boxes = [[int(x), int(y), int(bw), int(bh)] for (x, y, bw, bh) in rects]
        scores = [float(s) for s in (weights.flatten() if hasattr(weights, "flatten") else weights)]
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.nms_threshold)
        keep = [int(i) for i in (keep.flatten() if hasattr(keep, "flatten") else keep)]

        out = []
        for i in keep:
            x, y, bw, bh = boxes[i]
            box = (x / ratio, y / ratio, (x + bw) / ratio, (y + bh) / ratio)
            out.append(Detection(box, scores[i]))
        return out


def make_detector(backend, **kwargs):
    backend = backend.lower()
    if backend == "yolo":
        return YoloPersonDetector(**kwargs)
    if backend == "hog":
        return HogPersonDetector(**kwargs)
    raise ValueError("unknown detector backend %r (use 'yolo' or 'hog')" % backend)
