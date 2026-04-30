from ultralytics import YOLO

# Pretrained YOLOv8 pose model load karo
model = YOLO("yolov8n-pose.pt")  # ya yolov8s-pose.pt agar GPU strong hai

# Model train karo
model.train(
    data="C:/Users/Laptronics.co/OneDrive/Desktop/Poses Detection/dataset/dataset.yaml",
    epochs=50,         # jitni der tak train karna chahte ho
    imgsz=640,         # image size
    batch=8,           # GPU ke hisaab se adjust kar sakte ho
    lr0=0.01,          # learning rate
    pretrained=True,   # pretrained weights ka use ho raha hai
    plots=False
)
