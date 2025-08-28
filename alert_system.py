# alert_system.py
import smtplib
import cv2
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st  # ✅ add this

class AlertSystem:
    def __init__(self, email, password, recipient):
        self.email = email
        self.password = password
        self.recipient = recipient

    def send_alert(self, frame):
        # Save intruder snapshot
        img_path = "assets/intruder.jpg"
        cv2.imwrite(img_path, frame)

        # --- Email Setup ---
        subject = "🚨 Intruder Alert!"
        body = "An intruder has been detected by your Smart Security System."

        msg = MIMEMultipart()
        msg["From"] = self.email
        msg["To"] = self.recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with open(img_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=intruder.jpg")
        msg.attach(part)

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.email, self.password)
            server.sendmail(self.email, self.recipient, msg.as_string())
            server.quit()

            # ✅ Show Streamlit alert popup
            st.error("🚨 Intruder detected! Email sent with snapshot.")

        except Exception as e:
            st.warning(f"⚠️ Alert failed: {e}")
