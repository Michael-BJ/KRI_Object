import cv2
from PIL import Image
from ultralytics import YOLO
import time

# Load a pretrained YOLOv8n model
model = YOLO('416_30_v8.pt')

# Load image
frame = cv2.imread('korban.jpg')

# Run inference on the loaded image
results = model(frame, conf=0.9)[0]  # results list

status = 1 if len(results.boxes) > 0 else 0

for r in results:
    frame = r.plot()

fps_disp = "FPS: N/A"  # Since it's not real-time, FPS is not applicable

frame_resize = cv2.resize(frame, (640,640))
# Display the resulting frame
cv2.imshow('frame', frame_resize)
print("Status:", status)

# Wait for any key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()