# 🕺 Pose Detection using YOLOv8 + Django

A complete web application for human pose detection using custom trained YOLOv8 model. Users can upload images or use their camera for real-time pose detection.

## ✨ Features

- 📸 **Image Upload** - Upload any image and get pose keypoints detected
- 🎥 **Real-time Camera** - Live pose detection using your webcam
- 🎯 **Key Points Detection** - Shows all pose keypoints with connections
- 🖼️ **Annotated Output** - Returns image with pose drawn on it

## 🛠️ Tech Stack

- **Backend:** Django 5.x
- **ML Model:** YOLOv8 (Custom trained for pose detection)
- **Frontend:** HTML, CSS, JavaScript
- **Computer Vision:** OpenCV, Ultralytics

## 📁 Project Structure
Cv_Project/
├── myproject/ # Django project
│ ├── myapp/ # Main application
│ ├── model/ # Custom YOLOv8 model
│ ├── media/ # Uploaded images
│ └── templates/ # HTML templates
├── requirements.txt # Dependencies
└── README.md # This file


## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/Abdul-Hanan176/pose-detection-django.git
cd pose-detection-django
cd myproject 


### Step 2: Create virtual environment
```bash
python -m venv env
env\Scripts\activate  # For Windows

### Step 3: Install dependencies
pip install -r requirements.txt

###Step 4: Run the server
python manage.py runserver

Step 5: Open browser
Visit: http://127.0.0.1:8000



📞 Contact
Developer: Abdul Hanan
GitHub: Abdul-Hanan176

⭐ Show your support
Give a ⭐ if you like this project!
