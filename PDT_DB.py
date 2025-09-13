import cv2
from ultralytics import YOLO
import pymongo
from datetime import datetime

# MongoDB Setup (hardcoded)
mongo_uri = 'mongodb+srv://Maithri:Maithri_2025@cluster0.vefa5u1.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(mongo_uri)
db = client['PDT_DB']        #Database name
detections_collection = db['Collection_01']            #collection name

# Load YOLOv8 Medium Model for better accuracy
model = YOLO("yolov8m.pt")

# Choose Input Source
source = input("Enter 'camera' for webcam or 'video' for video file: ").strip().lower()

if source == 'video':
    video_path = input("Enter full path to video file: ").strip()
    cap = cv2.VideoCapture(video_path)
elif source == 'camera':
    cap = cv2.VideoCapture(0)
else:
    print("Invalid input. Exiting.")
    exit(1)

if not cap.isOpened():
    print("Error: Cannot open video source")
    exit(1)

print("Starting video stream... Press 'q' to quit.")

cv2.namedWindow('Pedestrian Detection & MongoDB', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pedestrian Detection & MongoDB', 800, 600)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch frame.")
        break

    frame_id += 1
    timestamp = datetime.now()

    # Run YOLO detection (no stream=True)
    results = model(frame)

    print(f"Frame {frame_id}: Detected {len(results[0].boxes)} total boxes")  # Debug print

    det_list = []
    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        # Lowered confidence threshold to 0.2 for more detections
        if cls == 0 and conf > 0.2:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_list.append({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'bbox': [x1, y1, x2, y2],
                'confidence': conf
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if det_list:
        detections_collection.insert_many(det_list)
        print(f"Inserted {len(det_list)} pedestrian detections for frame {frame_id}")
    else:
        print(f"No pedestrian detections in frame {frame_id}")

    cv2.putText(frame, f"Pedestrians: {len(det_list)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Pedestrian Detection & MongoDB', frame)

    key = cv2.waitKey(3000) & 0xFF  
    if key == ord('q'):
        print("Quitting on user request.")
        break

cap.release()
cv2.destroyAllWindows()
client.close()
