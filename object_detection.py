# object_detector.py
from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLOv8 pretrained model
        self.model = YOLO(model_path)

    def detect_from_webcam(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Run YOLOv8 detection
            results = self.model.predict(frame)

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Show result
            cv2.imshow("Webcam Object Detection", annotated_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ObjectDetector("yolov8n.pt")  # you can use yolov8s.pt, yolov8m.pt for better accuracy
    detector.detect_from_webcam()
