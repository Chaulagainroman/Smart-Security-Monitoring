from ultralytics import YOLO
import cv2

class HumanDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        people_boxes = []
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box)
                people_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return frame, people_boxes
