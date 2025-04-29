import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO  # Requires torch and ultralytics package

# Initialize YOLOv8 model (small version for speed)
model = YOLO('yolov8n.pt')
CLASS_NAMES = model.names
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes: car, bike, bus, truck

# Virtual detection lines for each lane (x1,y1,x2,y2)
DETECTION_LINES = {
    'north': [(200, 600), (400, 600)],
    'south': [(600, 200), (800, 200)],
    'east': [(1000, 400), (1000, 600)],
    'west': [(200, 200), (200, 400)]
}

# Traffic light control parameters
MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 60
BASE_GREEN_TIME = 10

# State variables
vehicle_counts = defaultdict(int)
current_phase = 'north'
phase_timer = 0

def draw_ui(frame):
    """Draw detection lines and traffic light status"""
    for lane, (start, end) in DETECTION_LINES.items():
        color = (0, 255, 0) if lane == current_phase else (0, 0, 255)
        cv2.line(frame, start, end, color, 2)
    return frame

def count_vehicles(detections):
    """Count vehicles crossing detection lines using basic box-line intersection"""
    counts = defaultdict(int)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        # print(x1,y1,x2,y2)
        center = ((x1 + x2)//2, (y1 + y2)//2)
        
        for lane, (start, end) in DETECTION_LINES.items():
            if point_in_line(center, start, end):
                counts[lane] += 1
                
    return counts

def point_in_line(point, line_start, line_end):
    """Check if point is near a line segment"""
    x, y = point
    (x1, y1), (x2, y2) = line_start, line_end
    distance = np.abs((y2 - y1)*x - (x2 - x1)*y + x2*y1 - y2*x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    # print(distance)
    return distance < 500  # 5 pixel threshold

def update_traffic_light():
    """Determine which lane gets priority"""
    global current_phase, phase_timer
    
    if phase_timer > 0:
        phase_timer -= 1
        return
    
    # Simple priority: lane with most vehicles
    max_count = max(vehicle_counts.values())
    candidates = [lane for lane, count in vehicle_counts.items() if count == max_count]
    
    if current_phase not in candidates or max_count == 0:
        current_phase = candidates[0] if candidates else 'north'
    
    # Calculate green time (clamped to min/max)
    green_time = BASE_GREEN_TIME + vehicle_counts[current_phase] * 2
    phase_timer = min(max(green_time, MIN_GREEN_TIME), MAX_GREEN_TIME)
    
    print(f"Changing to {current_phase} phase for {phase_timer} seconds")
    vehicle_counts.clear()

# Main processing loop
cap = cv2.VideoCapture('./traffic.mp4')  # Input video source
if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Vehicle detection
    results = model.track(frame, persist=True, verbose=False)
    
    # Process detections
    detections = []
    for box in results[0].boxes:
        if int(box.cls) in VEHICLE_CLASSES and box.conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({'bbox': (x1, y1, x2, y2)})
    # print(detections)
    # Update counts and UI
    counts = count_vehicles(detections)
    # print(counts)
    for lane, count in counts.items():
        vehicle_counts[lane] += count
    # print(vehicle_counts)    
    frame = draw_ui(frame)
    # print(frame)
    update_traffic_light()
    
    # Display
    cv2.imshow('Traffic Control', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()