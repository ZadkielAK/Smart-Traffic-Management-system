import serial
import time
import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# Output Directory and File Naming
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"traffic_output_{timestamp}.mp4")

# Connect to Arduino (⚠️ Update 'COM3' to your actual port if needed)
arduino = serial.Serial('COM3', 9600)
time.sleep(2)  # Allow Arduino to initialize

# Load YOLO Model
model = YOLO("runs/detect/traffic_model_finetuned4/weights/best.pt").to('cuda')

# ROI and Class Config
LANE_NAME = "Lane 1"
LANE_ROI = (480, 200, 960, 540)
TARGET_CLASSES = ['car', 'two_wheeler', 'auto', 'truck', 'bus']
CLASS_WEIGHTS = {'car': 1, 'two_wheeler': 0.5, 'auto': 1, 'truck': 2, 'bus': 2}
counted_ids = set()
total_counts = {cls: 0 for cls in TARGET_CLASSES}

# Adaptive Signal Timing Logic
def adaptive_signal_time(vehicle_count):
    base = 60
    if vehicle_count >= 40:
        return min(120, base + ((vehicle_count - 40) // 10 + 1) * 10)
    elif 30 <= vehicle_count < 40:
        return max(20, base - 20)
    else:
        return max(20, base - ((40 - vehicle_count - 1) // 10 + 1) * 10)

# Load Video
video_path = r"C:\Users\Anuraag\OneDrive\Documents\Desktop\ivp project\traffic_video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Start RED phase (turn on RED LED immediately)
arduino.write(b'R')

# RED PHASE ANALYSIS — First 20s
frame_id = 0
red_analysis_duration = 20  # seconds
analysis_frame_limit = int(red_analysis_duration * fps)
last_frame = None

while cap.isOpened() and frame_id < analysis_frame_limit:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    last_frame = frame.copy()

    results = model.track(frame, persist=True, device='cuda')[0]

    for box in results.boxes:
        if box.conf.item() < 0.6:
            continue
        cls_id = int(box.cls.item())
        cls_name = model.names[cls_id]
        if cls_name not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        lx1, ly1, lx2, ly2 = LANE_ROI
        if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id.item())
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_counts[cls_name] += 1

    lx1, ly1, lx2, ly2 = LANE_ROI
    time_passed = int(frame_id / fps)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 255, 0), 2)
    cv2.putText(frame, f"RED Phase | Counting: {time_passed}s", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicles Detected: {sum(total_counts.values())}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
    out.write(frame)
    cv2.imshow("Smart Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Compute adaptive times
weighted_total = sum(total_counts[cls] * CLASS_WEIGHTS[cls] for cls in TARGET_CLASSES)
red_time = adaptive_signal_time(weighted_total)
green_time = red_time

print(f"\n✅ Adaptive Red Time: {red_time} s")
print(f"✅ Adaptive Green Time: {green_time} s")
print(f"🚗 Detected Vehicle Count: {total_counts}")
print(f"🟢 Simulating Green Light Phase...")

# 🔴 RED PHASE EXTENSION
remaining_red_frames = int((red_time - red_analysis_duration) * fps)

for i in range(remaining_red_frames):
    frame = last_frame.copy()
    lx1, ly1, lx2, ly2 = LANE_ROI
    elapsed = red_analysis_duration + int(i / fps)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 255, 0), 2)
    cv2.putText(frame, f"🔴 RED Phase | {elapsed}s/{red_time}s", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicles Detected: {sum(total_counts.values())}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
    out.write(frame)
    cv2.imshow("Smart Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🟢 GREEN PHASE
arduino.write(b'G')  # Turn RED off, GREEN on
green_frames = int(green_time * fps)

for i in range(green_frames):
    frame = last_frame.copy()
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
    cv2.putText(frame, f" GREEN Phase | {i // int(fps)}s/{green_time}s", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    cv2.putText(frame, f"Vehicles Detected: {sum(total_counts.values())}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
    cv2.putText(frame, f" Red Timer was: {red_time}s", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 200), 2)
    out.write(frame)
    cv2.imshow("Smart Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ⚫ Turn OFF all LEDs
arduino.write(b'O')  # Custom code to reset LEDs

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("🎥 Simulation complete and output saved.")
