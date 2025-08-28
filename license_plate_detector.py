# license_plate_detector.py
from ultralytics import YOLO
import cv2

class LicensePlateDetector:
    def __init__(self, model_path="license_plate.pt"):
        # Load pretrained YOLO model (trained on license plates)
        self.model = YOLO(model_path)

    def detect_from_webcam(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = self.model.predict(frame, verbose=False)

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Show webcam feed
            cv2.imshow("License Plate Detection", annotated_frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = LicensePlateDetector("license_plate.pt")  # replace with your downloaded weights
    detector.detect_from_webcam()
