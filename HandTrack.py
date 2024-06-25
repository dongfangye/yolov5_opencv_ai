import cv2
import mediapipe as mp
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import torch
import pyautogui
from collections import deque

from umi_ocr import ocr_picture
from chattts import txt_to_audio

# -------------------------------------------------yolov5模型加载区-----------------------------------------------
# 原态yolov5s模型
model_path = 'weights/yolov5s.pt'  # 替换为你的本地模型路径
model = torch.hub.load('yolov5-6.0', 'custom', path=model_path, source='local')
# 高灿手指识别模型
model_finger_path = "weights/finger.pt"
model_finger = torch.hub.load("yolov5-6.0", "custom", path=model_finger_path, source="local")
# -------------------------------------------------yolov5截至区-----------------------------------------------

# 创建全局线程池
executor = ThreadPoolExecutor(max_workers=10)

# ----------------------------------------------------功能模块存放区---------------------------------------------
# 调用yolov5模型进行对象检测
def load_model(image_path: str):
    res = list()
    try:
        frame = cv2.imread(image_path)# 读取图像
        if frame is None:
            raise ValueError(f"Image at {image_path} could not be read")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# 将图像转换为RGB
        results = model(img)# 使用YOLOv5模型进行对象检测
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

# 截图 + 调用YOLOv5识别
def screenshot(x1, y1, x2, y2, img) -> None:
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cropped = img[y1:y2, x1:x2]
    cropped_path = 'tmp/screenshot.jpg' # 截图保存路径名称
    cv2.imwrite(cropped_path, cropped)
    print("Screenshot taken!")
    executor.submit(load_model, "tmp/screnshot.jpg")# 异步调用yolov5识别图片, 避免阻塞主程序

# 判断手是否处于稳定状态
def is_hand_stable(current_pos, prev_pos, threshold=10):
    if prev_pos is None:
        return False
    return np.linalg.norm(np.array(current_pos) - np.array(prev_pos)) < threshold

# 文字转语音, txt_to_audio函数中有阻塞等待, 建议使用异步调用
def text_to_speech(text: str) -> None:
    txt_to_audio(text)
    

# 计算手指数量
def count_fingers(hand_landmarks) -> int:
    fingers = []# 检查每个手指是否伸出
    # 大拇指, 大拇子不同于其他手指, 需要单独设置判断函数
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    if thumb_tip.x < thumb_ip.x:
        fingers.append(1)
    else:
        fingers.append(0)
    # 其余手指
    for idx, landmark in enumerate([mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                                    mp_hands.HandLandmark.RING_FINGER_TIP, 
                                    mp_hands.HandLandmark.PINKY_TIP]):
        finger_tip = hand_landmarks.landmark[landmark]
        finger_pip = hand_landmarks.landmark[landmark - 2]
        if finger_tip.y < finger_pip.y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)
# 鼠标平滑移动算法
def smooth_coordinates(x, y):
    x_deque.append(x)
    y_deque.append(y)
    smooth_x = int(np.mean(x_deque))
    smooth_y = int(np.mean(y_deque))
    return smooth_x, smooth_y
# ----------------------------------------------------功能模块截至此---------------------------------------------

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
is_clicking = False

cap = cv2.VideoCapture(0)

prev_positions = [None, None]
stable_counts = [0, 0]
stability_threshold = 30  # 稳定帧数阈值
max_stable_count = 60  # 最大稳定帧数
circle_radii = [0, 0]
max_circle_radius = 30

last_screenshot_time = 0
screenshot_interval = 5  # 截图间隔（秒）

# 鼠标平滑算法: 平滑参数
smooth_factor = 5
x_deque = deque(maxlen=smooth_factor)
y_deque = deque(maxlen=smooth_factor)

# 主循环, 模块调用
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_positions = [None, None]

# -------------------------------------------代码功能区----------------------------------------------------
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            # 获取食指和中指指尖位置
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            current_positions[hand_no] = (cx, cy)
            # 计算食指和中指指尖之间的距离
            distance = ((index_finger_tip.x - middle_finger_tip.x)**2 + (index_finger_tip.y - middle_finger_tip.y)**2)**0.5
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)
            
            smooth_x, smooth_y = smooth_coordinates(x, y)# 平滑坐标
            pyautogui.moveTo(smooth_x, smooth_y)# 移动鼠标
            
            # 判断手是否处于稳定状态
            if is_hand_stable(current_positions[hand_no], prev_positions[hand_no]):
                stable_counts[hand_no] = min(stable_counts[hand_no] + 1, max_stable_count)
                circle_radii[hand_no] = min(int(stable_counts[hand_no] / 2), max_circle_radius)
                cv2.circle(image, (cx, cy), circle_radii[hand_no], (0, 0, 255), 2)
            else:
                stable_counts[hand_no] = 0
                circle_radii[hand_no] = 0

            # 检测食指和中指是否并拢
            if distance < 0.05:
                if not is_clicking:
                    pyautogui.mouseDown()
                    is_clicking = True
            else:
                if is_clicking:
                    pyautogui.mouseUp()
                    is_clicking = False
                    
            # 计算伸出的手指数量
            num_fingers = count_fingers(hand_landmarks) # 后续调用yolov5模型取代此处识别
            #print(5-num_fingers)# 横平放置计算

            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 判断两手食指指尖是否处于稳定状态
    current_time = time.time()
    if all(pos is not None for pos in current_positions) and all(count >= stability_threshold for count in stable_counts):
        if current_time - last_screenshot_time > screenshot_interval:
            x1, y1 = current_positions[0]
            x2, y2 = current_positions[1]
            executor.submit(screenshot, x1, y1, x2, y2, image.copy())# 调用截图程序
            last_screenshot_time = current_time
    

    # 判断单手是否处于稳定状态, 且如果判断过双手处于稳定状态则跳过单手稳定判断
    elif all(pos is not None for pos in current_positions[:1]) and all(count >= stability_threshold for count in stable_counts[:1]):
        if current_time - last_screenshot_time > screenshot_interval:
            cv2.imwrite('tmp/ocr.jpg', image)# 获取当前摄像头图片并保存为ocr.jpg
            print("OCR image saved!")
            executor.submit(ocr_picture, "tmp/ocr.jpg")# 线程调用ocr识别
            last_screenshot_time = current_time
    else:
        pass


# -----------------------------------------功能区截至此-------------------------------------------------------

    # 更新上一帧位置
    for i in range(2):
        prev_positions[i] = current_positions[i] if current_positions[i] else None

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()