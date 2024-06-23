import cv2
import mediapipe as mp
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import torch

from umi_ocr import ocr_picture
import HandTrackingModule as htm


detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]  # 手指的关键点索引
totalFingers = 0  # 总手指数

# 全局加载YOLOv5模型
model_path = 'yolov5s.pt'  # 替换为你的本地模型路径
model = torch.hub.load('E:/Software/PythonProject/yolo/yolov5-6.0', 'custom', path=model_path, source='local')

def load_model(image_path: str):
    res = list()
    try:
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
                print(f"Detected {label} with confidence {row[4]:.2f}")   
                res.append(label)
    except Exception as e:
        print(f"Error in load_model: {e}")
    return res

# 创建全局线程池
executor = ThreadPoolExecutor(max_workers=10)

def screenshot(x1, y1, x2, y2, img):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cropped = img[y1:y2, x1:x2]
    cropped_path = 'test.jpg'
    cv2.imwrite(cropped_path, cropped)
    print("Screenshot taken!")

    # 异步调用yolov5识别图片, 避免阻塞主程序
    executor.submit(load_model, "test.jpg")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_positions = [None, None]
stable_counts = [0, 0]
stability_threshold = 30  # 稳定帧数阈值
max_stable_count = 60  # 最大稳定帧数
circle_radii = [0, 0]
max_circle_radius = 30

last_screenshot_time = 0
screenshot_interval = 5  # 截图间隔（秒）

def is_hand_stable(current_pos, prev_pos, threshold=10):
    if prev_pos is None:
        return False
    return np.linalg.norm(np.array(current_pos) - np.array(prev_pos)) < threshold

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_positions = [None, None]

    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            current_positions[hand_no] = (cx, cy)

            if is_hand_stable(current_positions[hand_no], prev_positions[hand_no]):
                stable_counts[hand_no] = min(stable_counts[hand_no] + 1, max_stable_count)
                circle_radii[hand_no] = min(int(stable_counts[hand_no] / 2), max_circle_radius)
                cv2.circle(image, (cx, cy), circle_radii[hand_no], (0, 0, 255), 2)
            else:
                stable_counts[hand_no] = 0
                circle_radii[hand_no] = 0

            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # 判断两手食指指尖是否处于稳定状态
    current_time = time.time()
    if all(pos is not None for pos in current_positions) and all(count >= stability_threshold for count in stable_counts):
        if current_time - last_screenshot_time > screenshot_interval:
            x1, y1 = current_positions[0]
            x2, y2 = current_positions[1]
            # 调用截图程序, 优化此处, 异步调用, 线程处理
            executor.submit(screenshot, x1, y1, x2, y2, image.copy())
            last_screenshot_time = current_time
    

    # 判断单手是否处于稳定状态, 且如果判断过双手处于稳定状态则跳过单手稳定判断
    elif all(pos is not None for pos in current_positions[:1]) and all(count >= stability_threshold for count in stable_counts[:1]):
         # 获取当前摄像头图片并保存为ocr.jpg
        if current_time - last_screenshot_time > screenshot_interval:
            cv2.imwrite('ocr.jpg', image)
            print("OCR image saved!")
            # 线程调用ocr识别
            executor.submit(ocr_picture, "ocr.jpg")
            last_screenshot_time = current_time
    else:
        #print("Not stable")
        pass


    for i in range(2):
        prev_positions[i] = current_positions[i] if current_positions[i] else None

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()