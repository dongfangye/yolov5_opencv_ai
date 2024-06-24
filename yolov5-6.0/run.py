import cv2
import torch
import numpy as np

# 加载本地训练好的YOLOv5模型
model_path = r'E:\Software\PythonProject\best.pt'  # 替换为你的本地模型路径
model = torch.hub.load('E:/Software/PythonProject/yolo/yolov5-6.0', 'custom', path=model_path, source='local')

# 获取摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将帧转换为RGB
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
            x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, model.names[int(labels[i])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        
    # 显示结果帧
    cv2.imshow('YOLOv5 Detection', frame)
    
    # 按'Q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
