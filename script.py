import cv2
import boto3
from twilio.rest import Client
import time
import threading
import queue
import numpy as np
import os
import csv
from datetime import datetime

# ===== CONFIGURATION =====
AWS_REGION = "us-east-1"
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', "AC322d8324975559aa1e426a3a28310042")
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', "b5a1e93ef1b60dda6f5eb19cd76d29ff")
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', "+12207585541")
EMERGENCY_PHONE_NUMBER = os.getenv('EMERGENCY_PHONE_NUMBER', "+918688754307")

# ===== RECORDING & LOGGING CONFIGURATION =====
RECORD_VIDEO = True
RECORD_ONLY_THREATS = True
VIDEO_SAVE_DIR = "recordings"
MAX_RECORDING_MINUTES = 5
LOG_FILE = "security_logs.csv"

# ===== PERFORMANCE OPTIMIZATION =====
PROCESSING_WIDTH = 320
PROCESSING_HEIGHT = 240
DETECTION_FREQUENCY = 0.5
USE_PARALLEL_PROCESSING = True
MAX_DETECTION_THREADS = 10
ENABLE_FRAME_SKIPPING = True
MAX_FRAME_QUEUE = 20
USE_GPU_ACCELERATION = False

# Video settings
VIDEO_SOURCE = 0  # 0 for webcam or "rtsp://..." for IP camera
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
TARGET_FPS = 30

# Detection settings
WEAPON_LABELS = ["Gun", "Knife"]
WEAPON_CONFIDENCE = 85
DETECTION_INTERVAL = 0.3

# ===== SYSTEM COMPONENTS =====
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.frame = frame if ret else None
    
    def read(self):
        with self.lock:
            return self.frame
    
    def stop(self):
        self.stopped = True
        self.cap.release()

class SecurityRecorder:
    def __init__(self):
        self.writer = None
        self.video_writer = None
        self.recording_start = None
        self.current_file = None
        
        os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if os.stat(LOG_FILE).st_size == 0:
                writer.writerow(["Timestamp", "Event", "Threat Type", "Confidence"])

    def start_recording(self, frame, threat_info=None):
        if not RECORD_VIDEO:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = f"{VIDEO_SAVE_DIR}/recording_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # Default FPS
        self.video_writer = cv2.VideoWriter(
            self.current_file, 
            fourcc, 
            fps, 
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )
        self.recording_start = time.time()
        self.log_event("Recording Started", threat_info)
        
    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.log_event("Recording Stopped")
            self.current_file = None

    def write_frame(self, frame):
        if self.video_writer:
            if (time.time() - self.recording_start) > MAX_RECORDING_MINUTES * 60:
                self.stop_recording()
                self.start_recording(frame)
            self.video_writer.write(frame)

    def log_event(self, event, threat_info=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if threat_info:
                writer.writerow([timestamp, event, threat_info['type'], threat_info['confidence']])
            else:
                writer.writerow([timestamp, event, "", ""])

class ThreatDetector:
    def __init__(self):
        self.rekognition = boto3.client("rekognition", region_name=AWS_REGION)
        self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.last_detection_time = 0
        self.last_alert_time = 0
        self.alert_cooldown = 30
        self.threats = []
        self.lock = threading.Lock()
    
    def detect(self, frame):
        current_time = time.time()
        if current_time - self.last_detection_time < DETECTION_INTERVAL:
            return
        
        self.last_detection_time = current_time
        
        def detection_task():
            _, img_encoded = cv2.imencode(".jpg", frame)
            response = self.rekognition.detect_labels(
                Image={"Bytes": img_encoded.tobytes()},
                MinConfidence=WEAPON_CONFIDENCE
            )
            
            current_threats = []
            for label in response["Labels"]:
                if label["Name"] in WEAPON_LABELS:
                    current_threats.append({
                        'type': label['Name'],
                        'confidence': float(label['Confidence'])
                    })
            
            with self.lock:
                self.threats = current_threats
                if current_threats and (current_time - self.last_alert_time) > self.alert_cooldown:
                    self.trigger_alert()
                    self.last_alert_time = current_time
        
        threading.Thread(target=detection_task, daemon=True).start()
    
    def trigger_alert(self):
        try:
            call = self.twilio_client.calls.create(
                url="http://demo.twilio.com/docs/voice.xml",
                to=EMERGENCY_PHONE_NUMBER,
                from_=TWILIO_PHONE_NUMBER
            )
            print(f"🚨 ALERT: Threat detected! Call SID: {call.sid}")
        except Exception as e:
            print(f"❌ Alert failed: {str(e)}")
    
    def get_threats(self):
        with self.lock:
            return self.threats.copy()

# ===== MAIN APPLICATION =====
def main():
    print("🚀 Starting security system...")
    
    vs = VideoStream(VIDEO_SOURCE).start()
    detector = ThreatDetector()
    recorder = SecurityRecorder()
    
    while vs.read() is None:
        time.sleep(0.1)
    
    try:
        while True:
            start_time = time.time()
            frame = vs.read()
            if frame is None:
                break
            
            detector.detect(frame)
            display_frame = frame.copy()
            
            fps = 1 / (time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            threats = detector.get_threats()
            if threats:
                cv2.putText(display_frame, "THREAT DETECTED!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                recorder.start_recording(frame, threats[0])
            
            try:
                cv2.imshow("Security Feed", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass
            
            if RECORD_VIDEO and (not RECORD_ONLY_THREATS or threats):
                recorder.write_frame(frame)
                
    finally:
        vs.stop()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        recorder.stop_recording()

if __name__ == "__main__":
    main()