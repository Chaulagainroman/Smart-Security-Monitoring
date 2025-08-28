import streamlit as st
from camera import Camera
from motion_detector import MotionDetector
from human_detector import HumanDetector
from tracker import Tracker
from alert_system import AlertSystem
from ultralytics import YOLO
import easyocr
import cv2
import os
import time
import pandas as pd
from collections import Counter

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("Smart Security System")

# Ensure assets folder exists
os.makedirs("assets/intruders", exist_ok=True)
os.makedirs("assets/vehicles", exist_ok=True)

# ---------------------------
# Session state initialization
# ---------------------------
ss = st.session_state
if "vehicle_logs" not in ss:
    ss.vehicle_logs = []   # list of dicts: {ts, img_path, plate}
if "person_logs" not in ss:
    ss.person_logs = []
if "dog_logs" not in ss:
    ss.dog_logs = []
if "intruder_thumbnails" not in ss:
    ss.intruder_thumbnails = []
if "alert_log" not in ss:
    ss.alert_log = []
if "tracking_history" not in ss:
    ss.tracking_history = []
if "last_alert_time" not in ss:
    ss.last_alert_time = 0
if "last_plate_seen" not in ss:
    ss.last_plate_seen = {}  # plate -> last timestamp

# ---------------------------
# Initialize modules
# ---------------------------
camera = Camera(0)
motion_detector = MotionDetector()
human_detector = HumanDetector()
tracker = Tracker()
alert_system = AlertSystem(
    email="euphoriaa0024@gmail.com",
    password="ieulibjgtxrxcvme",  # App Password
    recipient="alisonburger720@gmail.com"
)

# Load YOLO models
coco_detector = YOLO("yolov8n.pt")         # COCO model (person, dog, car, etc.)
plate_detector = YOLO("license_plate.pt")  # License plate model (pretrained)
ocr_reader = easyocr.Reader(['en'])

# Utility: IoU for box matching
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}

# ---------------------------
# Sidebar navigation
# ---------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Live Feed", "Vehicle Log", "Objects Log", "Snapshots", "Tracking History", "Alerts Log"],
)
run = st.sidebar.checkbox("Run Security System", True)
alert_cooldown = st.sidebar.slider("Alert Cooldown (seconds)", 5, 60, 30)

# ---------------------------
# Live Feed Page
# ---------------------------
if page == "Live Feed":
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.container(border=True).empty()
    objects_placeholder = st.empty()  # compact summary like: person×2, dog×1
    table_container = st.container()  # vehicle table below video

    while run:
        frame = camera.get_frame()
        if frame is None:
            break

        base_frame = frame.copy()

        # -------- Motion & Human detection --------
        base_frame, motion = motion_detector.detect(base_frame)
        people_boxes_md = []
        if motion:
            base_frame, people_boxes_md = human_detector.detect(base_frame)

        # -------- COCO detection --------
        coco_res = coco_detector.predict(base_frame, verbose=False)[0]
        detected_labels = []
        coco_boxes = []  # (x1,y1,x2,y2,label,conf)
        for box in coco_res.boxes:
            cls_id = int(box.cls[0])
            label = coco_detector.names[cls_id]
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_labels.append(label)
            coco_boxes.append((x1, y1, x2, y2, label, conf))

        # Log persons/dogs
        ts_now = time.strftime("%Y-%m-%d %H:%M:%S")
        if any(lbl == "person" for lbl in detected_labels):
            ss.person_logs.append(ts_now)
        if any(lbl == "dog" for lbl in detected_labels):
            ss.dog_logs.append(ts_now)

        # -------- License plate detection --------
        vehicle_boxes = [b for b in coco_boxes if b[4] in VEHICLE_CLASSES]
        plate_entries_this_frame = []
        if vehicle_boxes:
            plate_res = plate_detector.predict(base_frame, verbose=False)[0]
            for pbox in plate_res.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                plate_box = (px1, py1, px2, py2)

                # match plate to nearest vehicle
                best_iou = 0.0
                best_vehicle = None
                for vx1, vy1, vx2, vy2, vlabel, vconf in vehicle_boxes:
                    score = iou(plate_box, (vx1, vy1, vx2, vy2))
                    if score > best_iou:
                        best_iou = score
                        best_vehicle = (vx1, vy1, vx2, vy2)

                if best_vehicle and best_iou >= 0.1:
                    vx1, vy1, vx2, vy2 = best_vehicle
                    car_crop = base_frame[max(vy1-10,0):min(vy2+10, base_frame.shape[0]),
                                          max(vx1-10,0):min(vx2+10, base_frame.shape[1])]
                    plate_crop = base_frame[max(py1,0):min(py2, base_frame.shape[0]),
                                            max(px1,0):min(px2, base_frame.shape[1])]
                    if plate_crop.size == 0:
                        continue

                    # OCR
                    ocr = ocr_reader.readtext(plate_crop)
                    texts = [t for (_, t, conf) in ocr if conf >= 0.5]
                    if texts:
                        plate_text = texts[0].upper().replace(" ", "")
                        now = time.time()
                        last = ss.last_plate_seen.get(plate_text, 0)
                        if now - last > 10:
                            ts = time.strftime("%Y-%m-%d %H:%M:%S")
                            img_path = f"assets/vehicles/vehicle_{int(now)}.jpg"
                            cv2.imwrite(img_path, car_crop)
                            ss.vehicle_logs.append({
                                "Date Time": ts,
                                "Car Image": img_path,
                                "License Plate": plate_text,
                            })
                            ss.last_plate_seen[plate_text] = now
                            plate_entries_this_frame.append((plate_box, plate_text))

        # -------- Alerts --------
        current_time = time.time()
        if people_boxes_md and (current_time - ss.last_alert_time > alert_cooldown):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            clean_frame = base_frame.copy()
            snapshot_path = f"assets/intruders/intruder_{int(current_time)}.jpg"
            cv2.imwrite(snapshot_path, clean_frame)

            ss.tracking_history.append(timestamp)
            ss.intruder_thumbnails.append((snapshot_path, timestamp))

            try:
                alert_system.send_alert(clean_frame)
                ss.alert_log.append(f"• {timestamp} ✅ Email sent")
            except Exception as e:
                ss.alert_log.append(f"• {timestamp} ⚠️ Email failed: {e}")

            ss.last_alert_time = current_time

        # -------- Draw overlays --------
        display = base_frame.copy()
        # motion/human detector boxes
        for (x, y, w, h) in people_boxes_md:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "person(md)", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # coco boxes
        for x1, y1, x2, y2, label, conf in coco_boxes:
            if label in ("person",):
                color = (0, 200, 0)
            elif label in ("dog", "cat"):
                color = (200, 120, 0)
            elif label in VEHICLE_CLASSES:
                color = (0, 150, 255)
            else:
                color = (180, 180, 180)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{label}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # plate boxes
        for (plate_box, plate_text) in plate_entries_this_frame:
            px1, py1, px2, py2 = plate_box
            cv2.rectangle(display, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(display, plate_text, (px1, max(0, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        display = tracker.update(display, people_boxes_md)

        # -------- Present --------
        video_placeholder.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB")

        if detected_labels:
            cnt = Counter(detected_labels)
            summary = ", ".join([f"{k}×{v}" for k, v in cnt.items()])
        else:
            summary = "None"
        objects_placeholder.write(f"**Detected Objects:** {summary}")

        # vehicle table
        with table_container:
            if ss.vehicle_logs:
                st.subheader("🚘 Detected Vehicles (recent)")
                header = st.columns([2, 2, 2])
                header[0].markdown("**Date Time**")
                header[1].markdown("**Car Image**")
                header[2].markdown("**License Plate**")
                for row in reversed(ss.vehicle_logs[-10:]):
                    cols = st.columns([2, 2, 2])
                    cols[0].write(row["Date Time"])
                    cols[1].image(row["Car Image"], width=120)
                    cols[2].write(row["License Plate"])
            else:
                st.info("No vehicles detected yet.")

# ---------------------------
# Vehicle Log Page
# ---------------------------
elif page == "Vehicle Log":
    st.subheader("🚘 Vehicle Detection Log")
    if ss.vehicle_logs:
        df = pd.DataFrame(ss.vehicle_logs)
        st.dataframe(df.drop(columns=["Car Image"]))
        for row in reversed(ss.vehicle_logs):
            cols = st.columns([2, 2, 2])
            cols[0].write(row["Date Time"])
            cols[1].image(row["Car Image"], width=120)
            cols[2].write(row["License Plate"])
        csv = pd.DataFrame(ss.vehicle_logs).drop(columns=["Car Image"]).to_csv(index=False)
        st.download_button("Download CSV", csv, "vehicle_logs.csv", "text/csv")
    else:
        st.info("No vehicle logs yet.")

# ---------------------------
# Objects Log Page
# ---------------------------
elif page == "Objects Log":
    st.subheader("📦 Objects Log (people & dogs)")
    col1, col2 = st.columns(2)
    col1.markdown("**Person events**")
    if ss.person_logs:
        col1.table({"Timestamp": list(reversed(ss.person_logs))})
    else:
        col1.info("No person events yet.")
    col2.markdown("**Dog events**")
    if ss.dog_logs:
        col2.table({"Timestamp": list(reversed(ss.dog_logs))})
    else:
        col2.info("No dog events yet.")

# ---------------------------
# Snapshots Page
# ---------------------------
elif page == "Snapshots":
    st.subheader("📸 Intruder Snapshots")
    if ss.intruder_thumbnails:
        cols = st.columns(3)
        for idx, (img_path, ts) in enumerate(reversed(ss.intruder_thumbnails)):
            with cols[idx % 3]:
                st.image(img_path, use_column_width=True, caption=f"Detected: {ts}")
    else:
        st.info("No intruder snapshots yet.")

# ---------------------------
# Tracking History Page
# ---------------------------
elif page == "Tracking History":
    st.subheader("📍 Tracking History")
    if ss.tracking_history:
        st.table({"Timestamp": list(reversed(ss.tracking_history))})
    else:
        st.info("No tracking events yet.")

# ---------------------------
# Alerts Log Page
# ---------------------------
elif page == "Alerts Log":
    st.subheader("⚠️ Alerts Log")
    if ss.alert_log:
        for log in reversed(ss.alert_log):
            st.write(log)
    else:
        st.info("No alerts yet.")

# Release camera
camera.release()
