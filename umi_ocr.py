import requests
import json
import base64
import os
from typing import Union

def ocr_picture(image_path) -> Union[str, None]:
    """
    调用Umi_OCR识别图片
    :param image_path: 图片路径
    :return: 识别结果
    """
    if os.path.exists(image_path) is False:
        print("图片不存在")
        return None
    with open(image_path, "rb") as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
    url = "http://127.0.0.1:1224/api/ocr" # API接口地址
    data = {
        "base64": base64_image,
        "options": {
            "ocr.language": "models/config_chinese.txt",
            "ocr.cls": False,
            "ocr.limit_side_len": 960,
            "tbpu.parser": "multi_para",
            "data.format": "text",
        }
    }
    headers = {"Content-Type": "application/json"}
    data_str = json.dumps(data)
    try:
        response = requests.post(url, data=data_str, headers=headers)
        if response.status_code == 200:
            res_dict = json.loads(response.text)
            res_dict = res_dict["data"]
            print("返回值字典\n", res_dict)
            return res_dict
        else:
            print("请求失败，状态码：", response.status_code)
    except Exception as e:
        print("请求异常：", e)

