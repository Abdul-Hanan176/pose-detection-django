from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from .models import PoseImage
from django.core.files.base import ContentFile
import uuid

MODEL_PATH = 'model/yolov8n-pose.pt'
model = YOLO(MODEL_PATH)

def save_pose_to_db(original_image_file, annotated_image_np, poses):
    # Convert annotated image to Django ImageField
    success, buffer = cv2.imencode('.jpg', annotated_image_np)
    if not success:
        raise ValueError("Cannot encode annotated image")
    
    annotated_image_file = ContentFile(buffer.tobytes(), name=f"{uuid.uuid4()}.jpg")
    
    pose_obj = PoseImage.objects.create(
        original_image=original_image_file,
        annotated_image=annotated_image_file,
        keypoints=poses
    )
    return pose_obj

def image_to_base64(image_np):
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("Could not encode image to JPEG")
    return base64.b64encode(buffer).decode('utf-8')

def process_image_for_poses(image_file):
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    results = model(image_np)
    annotated_img = results[0].plot()  # annotated image with skeleton
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    img_base64 = image_to_base64(annotated_img)

    poses = []
    if results[0].keypoints is not None:
        for k in results[0].keypoints.xy.cpu().numpy():
            poses.append({"keypoints": k.tolist()})

    return {"image": img_base64, "poses": poses}

def process_frame_for_poses(frame_blob):
    # Convert frame blob to numpy image
    image = Image.open(BytesIO(frame_blob.read())).convert("RGB")
    image_np = np.array(image)
    results = model(image_np)

    # Use plot() to get full skeleton image
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    img_base64 = image_to_base64(annotated_img)

    # Extract keypoints for JS overlay scaling
    poses = []
    if results[0].keypoints is not None:
        for k in results[0].keypoints.xy.cpu().numpy():
            poses.append({
                "keypoints": k.tolist()
            })

    # Return annotated image + keypoints
    return {"image": img_base64, "poses": poses}
