import cv2
import torch
import numpy as np

# 加载本地训练好的YOLOv5模型
model_path = 'yolov5s.pt'  # 替换为你的本地模型路径
# E:/Software/PythonProject/yolo/yolov5-6.0为yolov5文件夹安装路径
model = torch.hub.load('E:/Software/PythonProject/yolo/yolov5-6.0', 'custom', path=model_path, source='local')
def load_model(image_path:str):
    res = list()
    
    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Image at {image_path} could not be read")
    # 将图像转换为RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 使用YOLOv5模型进行对象检测
    results = model(img)
    # 解析检测结果，并将张量从GPU复制到CPU，再转换为NumPy数组
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    cords = results.xyxyn[0][:, :-1].cpu().numpy()
    # 绘制检测结果
    n = len(labels)
    for i in range(n):
        row = cords[i]
        if row[4] >= 0.2:  # 设置置信度阈值
            label = model.names[int(labels[i])] 
            res.append(label)
    return res
if __name__ == '__main__':
    image_path = 'app2.jpg'  # 替换为你的测试图片路径
    data = load_model(image_path)  
    print(data)
    