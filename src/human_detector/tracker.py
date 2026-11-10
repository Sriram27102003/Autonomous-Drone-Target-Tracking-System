"""Multi-object tracking by bounding-box overlap (IoU), plus target selection.

Boxes are ``(x1, y1, x2, y2)`` tuples in pixel coordinates.
"""


def iou(a, b):
    """Intersection-over-union of two boxes; 0.0 when they do not overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class Detection(object):
    __slots__ = ("box", "score")

    def __init__(self, box, score=1.0):
        self.box = tuple(float(v) for v in box)
        self.score = float(score)

    def __repr__(self):
        return "Detection(box=%r, score=%.2f)" % (self.box, self.score)


class Track(object):
    """One tracked person. ``box`` is smoothed to reduce jitter."""

    def __init__(self, track_id, detection):
        self.id = track_id
        self.box = detection.box
        self.score = detection.score
        self.hits = 1          # total frames matched
        self.misses = 0        # consecutive frames without a match
        self.age = 1           # frames since creation

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def area(self):
        return box_area(self.box)

    def __repr__(self):
        return "Track(id=%d, box=%r, hits=%d, misses=%d)" % (
            self.id, tuple(round(v, 1) for v in self.box), self.hits, self.misses)


class IoUTracker(object):
    """Greedy IoU tracker.

    Each frame, detections are matched to existing tracks in order of highest
    overlap. Unmatched detections start new tracks; tracks unmatched for more
    than ``max_misses`` frames are dropped. A track is *confirmed* (reported)
    once it has been matched ``min_hits`` times, which filters one-frame false
    positives.
    """

    def __init__(self, iou_threshold=0.3, max_misses=15, min_hits=3, smoothing=0.6):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.smoothing = smoothing  # weight of the new detection, 1.0 = no smoothing
        self.tracks = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        """Advance one frame. Returns confirmed tracks seen in this frame."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                overlap = iou(track.box, det.box)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._apply_match(self.tracks[ti], detections[di])

        for ti, track in enumerate(self.tracks):
            track.age += 1
            if ti not in matched_tracks:
                track.misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(Track(self._next_id, det))
                self._next_id += 1

        return self.confirmed_tracks()

    def confirmed_tracks(self):
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses == 0]

    def get(self, track_id):
        for t in self.tracks:
            if t.id == track_id:
                return t
        return None

    def _apply_match(self, track, det):
        a = self.smoothing
        track.box = tuple(a * n + (1.0 - a) * o for n, o in zip(det.box, track.box))
        track.score = det.score
        track.hits += 1
        track.misses = 0


class TargetSelector(object):
    """Picks which tracked person to follow and sticks with them.

    In auto mode the largest (usually the closest) confirmed track is chosen
    and kept until the tracker drops it. ``lock(track_id)`` forces a specific
    person; ``lock(None)`` returns to auto mode.
    """

    def __init__(self):
        self.target_id = None
        self.requested_id = None

    def lock(self, track_id):
        """Follow ``track_id``; ``None`` = auto mode (keeps the current target until lost)."""
        self.requested_id = track_id
        if track_id is not None:
            self.target_id = track_id

    def select(self, tracker, confirmed):
        """Return the track to follow this frame, or ``None``."""
        if self.target_id is not None:
            current = tracker.get(self.target_id)
            if current is not None:
                # Still alive. Only report it when it was seen this frame.
                return current if current.misses == 0 and current.hits >= tracker.min_hits else None
            if self.requested_id is not None:
                return None  # the requested person is gone; wait instead of switching
            self.target_id = None

        if not confirmed:
            return None
        best = max(confirmed, key=lambda t: t.area)
        self.target_id = best.id
        return best
