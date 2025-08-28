# Smart-Security-Monitoring

Smart Security Monitoring

An AI-powered smart security monitoring system that uses computer vision techniques for real-time human detection, motion tracking, pose estimation, and alert notifications. This project integrates OpenCV, YOLO, and Streamlit into a unified system for safety and security applications.

🚀 Features

👁️ Motion Detection – Detects unusual movements in surveillance footage.

🧍 Human Detection – Identifies and tracks people in the frame.

🕺 Pose Estimation – Tracks multiple human poses for activity monitoring.

🎯 Object Tracking – Follows detected subjects across frames.

📸 Intruder Snapshots – Captures and stores intruder images.

🔔 Smart Alerts – Notifies via integrated alert system when anomalies are detected.

🌐 Streamlit Dashboard – Real-time visualization and monitoring through a web-based UI.

🛠️ Tech Stack

Python 3.8+

OpenCV – Video and image processing

YOLO (Ultralytics) – Human/Object detection

MediaPipe – Pose detection and landmark tracking

Streamlit – Interactive monitoring dashboard

EasyOCR – (Optional) text recognition in frames

TensorFlow Lite – Optimized inference on CPU


git clone https://github.com/Chaulagainroman/Smart-Security-Monitoring.git
cd Smart-Security-Monitoring


python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac
