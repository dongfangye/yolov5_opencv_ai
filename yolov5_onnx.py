import onnxruntime
import numpy as np
import cv2

class YOLOv5ONNX:
    def __init__(self, model_path, input_size=(640, 640), conf_threshold=0.3):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image_path):
        """预处理输入图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, self.input_size)
        image_resized = image_resized.astype(np.float32) / 255.0
        image_resized = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
        image_resized = np.expand_dims(image_resized, axis=0)  # 添加batch维度
        return image, image_resized

    def postprocess(self, outputs, image):
        """后处理模型输出"""
        boxes, scores, class_ids = [], [], []
        detections = outputs[0]
        
        if detections is None or len(detections) == 0:
            print("No detections found in the output.")
            return boxes, scores, class_ids
        
        for detection in detections:
            if len(detection) >= 6:  # 确保检测结果包含至少6个值
                score = detection[4]
                if score > self.conf_threshold:  # 置信度
                    x_center, y_center, width, height = detection[0:4]
                    x_min = int((x_center - width / 2) * image.shape[1])
                    y_min = int((y_center - height / 2) * image.shape[0])
                    width = int(width * image.shape[1])
                    height = int(height * image.shape[0])
                    boxes.append([x_min, y_min, width, height])
                    scores.append(score)
                    class_ids.append(int(detection[5]))
        return boxes, scores, class_ids

    def detect(self, image_path):
        image, image_resized = self.preprocess(image_path)
        print("Preprocessed image shape:", image_resized.shape)  # 调试信息
        outputs = self.session.run([self.output_name], {self.input_name: image_resized})
        if not outputs or len(outputs) == 0:
            raise ValueError("Model output is empty")
        # print("Model outputs:", outputs)  # 调试信息
        boxes, scores, class_ids = self.postprocess(outputs, image)
        return image, boxes, scores, class_ids

    def draw_boxes(self, image, boxes, scores, class_ids):
        """在图像上绘制检测框"""
        for box, score, class_id in zip(boxes, scores, class_ids):
            x_min, y_min, width, height = box
            cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 2)
            cv2.putText(image, f'{class_id} {score:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

# 全局模型对象
model_path = 'yolov5s.onnx'
global_model = YOLOv5ONNX(model_path)

def detect_image(image_path):
    return global_model.detect(image_path)

def main(image_path):
    image, boxes, scores, class_ids = detect_image(image_path)
    if boxes:
        for box, score, class_id in zip(boxes, scores, class_ids):
            print(f'Box: {box}, Score: {score}, Class ID: {class_id}')
        image_with_boxes = global_model.draw_boxes(image, boxes, scores, class_ids)
        cv2.imshow('Detection', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No objects detected.")

if __name__ == '__main__':
    main('ocr.jpg')
