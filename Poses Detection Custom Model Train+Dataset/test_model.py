from ultralytics import YOLO
import cv2

# 1️⃣ Load trained YOLOv8 model (apna best.pt path use karo)
model = YOLO("runs/pose/train3/weights/best.pt")

# 2️⃣ Inference on ek single image ka exact path do 👇
results = model.predict(
    source=r"C:\Users\Laptronics.co\OneDrive\Desktop\Poses Detection\test1.jpg",  # <-- apni test image ka poora path
    show=True,      # window me result show karega
    conf=0.25,      # confidence threshold
    save=True,      # results ko save karega (runs/predict-pose/ me)
    imgsz=640       # image size
)

# 3️⃣ Optional: display processed frame
for result in results:
    img = result.plot()  # detections draw karta hai
    cv2.imshow("YOLOv8 Results", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
