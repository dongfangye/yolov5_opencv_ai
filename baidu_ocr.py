import requests  
import random  
import json  
import os  
from hashlib import md5  
import base64
import time
# 百度翻译图片文字识别
def translate_image(image_path, from_lang='zh', to_lang='en', app_id='####', app_key='####', cuid='APICUID', mac='mac'):  
    # 注意：这里假设有一个特定的API端点用于处理图片上传和翻译  
    # 您需要替换为实际的API端点  
    endpoint = 'https://fanyi-api.baidu.com/api/trans/sdk/picture'  
  
    # Generate salt and sign  
    def get_md5(string, encoding='utf-8'):  
        return md5(string.encode(encoding)).hexdigest()  
  
    def get_file_md5(file_name):  
        with open(file_name, 'rb') as f:  
            data = f.read()  
            return md5(data).hexdigest()  
  
    salt = random.randint(32768, 65536)  
    sign = get_md5(app_id + get_file_md5(image_path) + str(salt) + cuid + mac + app_key)  
  
    # Build request  
    payload = {  
        'from': from_lang,  
        'to': to_lang,  
        'appid': app_id,  
        'salt': salt,  
        'sign': sign,  
        'cuid': cuid,  
        'mac': mac,  
        # 如果有其他必需的表单字段，也在这里添加  
    }  
  
    # 使用files参数发送文件  
    files = {'image': open(image_path, 'rb')}  # 通常不需要指定文件名和内容类型，除非API有特殊要求  
  
    # Send request  
    response = requests.post(endpoint, data=payload, files=files)  
  
    # 检查响应状态码  
    if response.status_code == 200:  
        result = response.json()  
        if result["error_code"] == 0 or result["error_code"] == "0":
            res = list()
            zh_data = result["data"]["content"]
            for tmp in zh_data:
                tmp = tmp["src"]
                res.append(tmp)
            print(res)
            return res
    else:  
        print("Error:", response.status_code, response.text)  
        return None  

def get_file_content_as_base64(path, urlencoded=False):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
    return content

def get_picture_understanding_result(task_id, access_token):
    url = f"https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/get-result?access_token={access_token}"
    payload = json.dumps({
        "task_id":task_id
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    while True:
        if data["result"]["ret_msg"] == "success":
            res = data["result"]["description"]
            print(res)
            break
        elif data["result"]["ret_msg"] == "processing":
            time.sleep(3)
        else:
            break


# 百度图像内容理解
def image_understanding(image_path):  
    access_token = "#####"
    encoded_image = get_file_content_as_base64(image_path)
    url = f"https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/request?access_token={access_token}"
    payload = json.dumps({
        "image": encoded_image,
        "question":"图片里面有什么",
        "output_CHN": True
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.json()
    task_id = res["result"]["task_id"] 
    print(task_id)
    get_picture_understanding_result(task_id,access_token)

if __name__ == '__main__':  
    image_understanding("data/txt3.jpg")  