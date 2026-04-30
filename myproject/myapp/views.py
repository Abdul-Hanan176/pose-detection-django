from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .utils import process_image_for_poses, process_frame_for_poses, save_pose_to_db
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
import base64
import numpy as np
from PIL import Image


def pose_detection_page(request):
    return render(request, 'pose_detection.html')

@csrf_exempt
def detect_pose_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image'})
        
        result = process_image_for_poses(image_file)
        
        # Save to DB
        save_pose_to_db(original_image_file=image_file, 
                        annotated_image_np=np.array(Image.open(BytesIO(base64.b64decode(result['image'])))),
                        poses=result['poses'])
        
        return JsonResponse(result)

@csrf_exempt
def detect_pose_camera(request):
    if request.method == 'POST':
        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({'error': 'No frame'})
        result = process_frame_for_poses(frame_file)
        return JsonResponse(result)
    
