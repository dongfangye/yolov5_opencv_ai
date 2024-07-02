import time
import requests
import json
import base64
import urllib.parse
# 百度图片内容理解
def get_file_content_as_base64(path, urlencoded=False):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def Picture_Understand(image_path, question:str = None,token:str = None):
    encoded_image = get_file_content_as_base64(image_path)
    url = f"https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/request?access_token={token}"
    payload = json.dumps({
        "image": encoded_image,
        "question": question,
        "output_CHN": True
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    task_id = response.json()["result"]["task_id"]
    url = f"https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/get-result?access_token={token}"
    payload = json.dumps({
        "task_id":task_id
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    while True:
        if data["result"]["ret_msg"] == "successful":
            print(data)
            break
        elif data["result"]["ret_msg"] == "processing":
            print("Processing, please wait...")
        else:
            print(f"Unexpected status: {data['result']['ret_msg']}")
            break
        response = requests.post(url, headers=headers, data=payload)
        data = response.json()
        time.sleep(2)
    res = data["result"]["description"]
    return res

if __name__ == '__main__':
    token = "####"
    image_path = "4.jpg"
    question = "你好，请问这是一个什么样的图片？"
    data = Picture_Understand(image_path,question=question, token=token)
    print(data)