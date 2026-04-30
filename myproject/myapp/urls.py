# urls.py (in your app or project level)
from django.urls import path
from . import views  # Adjust import based on your app structure

urlpatterns = [
    # ... existing URLs ...
    path('pose-detection/', views.pose_detection_page, name='pose_detection_page'),
    path('detect-image/', views.detect_pose_image, name='detect_image'),
    path('detect-camera/', views.detect_pose_camera, name='detect_camera'),
]