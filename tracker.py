import cv2

class Tracker:
    def __init__(self):
        self.tracked_boxes = []

    def update(self, frame, boxes):
        """
        Update tracked objects using YOLO detection boxes.
        boxes format: [(x1, y1, x2, y2), ...]
        """
        self.tracked_boxes = boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame
