import base64
import urllib
import requests
# 百度通用物体场景识别
def get_file_content_as_base64(path, urlencoded=False):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def Picture_Search(image_path, token:str = None) -> list:
    image_base64 = get_file_content_as_base64(image_path, True)# 获取图片的base64编码

    url = f"https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general?access_token={token}"

    payload = f'image={image_base64}'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    res = list()
    for tmp in data["result"]:
        root = tmp["root"]
        keyword = tmp["keyword"]
        res.append([root, keyword])
    return res

if __name__ == '__main__':
    image_path = "2.jpg"
    Picture_Search(image_path,"")