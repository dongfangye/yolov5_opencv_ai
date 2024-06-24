import torch
import cv2
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 定义图像特征提取模型（如 ResNet50）
feature_extractor = models.resnet50(pretrained=True)
feature_extractor.eval()

# 定义图像预处理函数
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image).numpy().flatten()
    return features

def detect_and_crop(image_path, model, target_label):
    results = model(image_path)
    detections = results.pandas().xyxy[0]  # YOLOv5 检测结果

    cropped_images = []
    for _, row in detections.iterrows():
        if row['name'] == target_label:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            image = cv2.imread(image_path)
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
    return cropped_images

def compare_images(image1_features, image2_features):
    return np.linalg.norm(image1_features - image2_features)

# 示例：检测并裁剪目标对象
image_path = 'path/to/your/image.jpg'
target_label = 'your_target_label'
cropped_images = detect_and_crop(image_path, model, target_label)

# 对比文件夹中的图像
folder_path = 'path/to/your/folder'
threshold = 0.5  # 自定义阈值

for cropped_image in cropped_images:
    cv2.imwrite('temp_cropped_image.jpg', cropped_image)
    cropped_image_features = extract_features('temp_cropped_image.jpg')
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            folder_image_features = extract_features(image_path)
            similarity = compare_images(cropped_image_features, folder_image_features)
            if similarity < threshold:
                print(f'Found similar image: {filename} with similarity {similarity}')
