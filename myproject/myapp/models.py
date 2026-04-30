from django.db import models

# Create your models here.
from django.db import models

class PoseImage(models.Model):
    original_image = models.ImageField(upload_to='uploads/')
    annotated_image = models.ImageField(upload_to='annotated/')
    keypoints = models.JSONField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"PoseImage {self.id} - {self.uploaded_at}"
