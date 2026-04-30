from pycocotools.coco import COCO
import os

# Load your COCO JSON
coco = COCO(r"C:/Users/Laptronics.co/OneDrive/Desktop/Poses Detection/dataset/Pose_dataset/annotations/instances_train.json")
img_dir = r'C:\Users\Laptronics.co\OneDrive\Desktop\Poses Detection\dataset\Pose_dataset\images\train'
label_dir = r'C:\Users\Laptronics.co\OneDrive\Desktop\Poses Detection\dataset\Pose_dataset\labels\train'

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    with open(os.path.join(label_dir, f"{img_info['file_name'].split('.')[0]}.txt"), 'w') as f:
        for ann in anns:
            if 'keypoints' in ann:
                cls = ann['category_id'] - 1  # Adjust if needed
                bbox = ann['bbox']  # [x, y, w, h] - normalize
                kpts = ann['keypoints']  # [x1,y1,v1, x2,y2,v2, ...]
                # Normalize bbox and keypoints
                w, h = img_info['width'], img_info['height']
                bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
                kpts_norm = [kpts[i]/w if i % 3 == 0 else kpts[i]/h if i % 3 == 1 else kpts[i] for i in range(len(kpts))]
                line = f"{cls} {' '.join(map(str, bbox_norm))} {' '.join(map(str, kpts_norm))}\n"
                f.write(line)