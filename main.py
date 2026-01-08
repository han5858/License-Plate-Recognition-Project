import cv2
import easyocr
import pandas as pd
import Levenshtein
from ultralytics import YOLO
from datetime import datetime
import os

# --- CONFIGURATION --- 
VIDEO_PATH = "video.mp4"            # Input video file
MODEL_PATH = "plate_model.pt"       # Trained YOLO model
OUTPUT_VIDEO = "result_video.mp4"   # Processed output video
REPORT_FILE = "plate_report.xlsx"   # Excel report file
TARGET_PLATE = "34 IST 34"          # Sample authorized plate for matching

# --- SYSTEM CHECKS ---
if not os.path.exists(VIDEO_PATH):
    print(f"❌ ERROR: '{VIDEO_PATH}' not found! Please check the file name.")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: '{MODEL_PATH}' not found! Make sure the model is in the folder.")
    exit()

print("[INFO] Initializing AI System... ")

# 1. Load Model and OCR Reader
# Using GPU if available, otherwise CPU
reader = easyocr.Reader(['en'], gpu=True) 
model = YOLO(MODEL_PATH)

# 2. Video Capture Settings
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video Writer (MP4 format)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

detected_records = [] 
frame_count = 0

print("[INFO] Processing started! Press 'q' to stop manually.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video stream.")
        break

    frame_count += 1
    
    # OPTIONAL: Uncomment below to process every 3rd frame (Faster processing)
    # if frame_count % 3 != 0: continue

    # --- A. OBJECT DETECTION (YOLOv8) ---
    # conf=0.25 -> Detection threshold
    results = model(frame, verbose=False, conf=0.25)[0]

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Boundary checks to prevent crashing
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # --- B. CROP & OCR (Text Recognition) ---
        plate_crop = frame[y1:y2, x1:x2]

        try:
            # detail=0 returns only text list
            ocr_result = reader.readtext(plate_crop, detail=0)
            
            if ocr_result:
                # Cleanup text (Remove spaces, keep alphanumerics, uppercase)
                plate_text = ''.join(e for e in ocr_result[0] if e.isalnum()).upper()

                # Filter out noise (too short strings)
                if len(plate_text) > 4:
                    
                    # --- C. SIMILARITY ANALYSIS (Levenshtein) ---
                    # Check similarity with the Target Plate
                    similarity = Levenshtein.ratio(plate_text, TARGET_PLATE.replace(" ", ""))
                    
                    color = (0, 0, 255) # Red (Unauthorized / Unknown)
                    status = "UNKNOWN"
                    
                    # If similarity > 70%, mark as Authorized
                    if similarity > 0.7: 
                        color = (0, 255, 0) # Green
                        status = "ACCESS GRANTED"

                    # --- D. VISUALIZATION ---
                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw Label Background & Text
                    cv2.rectangle(frame, (x1, y1-40), (x2, y1), color, -1)
                    cv2.putText(frame, f"{plate_text}", (x1 + 5, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Log to Console
                    print(f"Frame {frame_count}: {plate_text} -> {status} (Conf: {score:.2f})")

                    # Append to Records List
                    detected_records.append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Frame_ID": frame_count,
                        "Detected_Plate": plate_text,
                        "Status": status,
                        "Confidence_Score": round(score, 2)
                    })

        except Exception as e:
            # Skip errors during OCR processing
            continue

    # --- E. SAVE & DISPLAY ---
    # Write processed frame to output video
    out.write(frame)

    # Resize for display (Fit to screen)
    display_frame = cv2.resize(frame, (1024, 600))
    cv2.imshow("AI License Plate Recognition System", display_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Process interrupted by user.")
        break

# --- F. CLEANUP ---
cap.release()
out.release()
cv2.destroyAllWindows()

# Export Report to Excel
if len(detected_records) > 0:
    df = pd.DataFrame(detected_records)
    df.to_excel(REPORT_FILE, index=False)
    print(f"\n✅ SUCCESS! Processing Complete.")
    print(f"📼 Output Video: {OUTPUT_VIDEO}")
    print(f"📊 Excel Report: {REPORT_FILE}")
else:
    print("\n⚠️ No plates detected in the video.")