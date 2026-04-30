from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("C:/Users/Laptronics.co/OneDrive/Desktop/Poses Detection/runs/pose/train3/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Optional: set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    # Run model inference (stream=True returns generator)
    for result in model.track(frame, stream=True):
        # Draw skeleton / keypoints on frame
        annotated_frame = result.plot()
        cv2.imshow("Pose Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
