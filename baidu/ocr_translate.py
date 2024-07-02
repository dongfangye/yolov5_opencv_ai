import requests  
import random  
from hashlib import md5  
# 百度翻译图片文字识别
def translate_image(image_path, from_lang='zh', to_lang='en', app_id='####', app_key='####', cuid='APICUID', mac='mac'):  
    if not image_path:
        return None
    endpoint = 'https://fanyi-api.baidu.com/api/trans/sdk/picture'  
    def get_md5(string, encoding='utf-8'):  
        return md5(string.encode(encoding)).hexdigest()  
    def get_file_md5(file_name):  
        with open(file_name, 'rb') as f:  
            data = f.read()  
            return md5(data).hexdigest()  
  
    salt = random.randint(32768, 65536)  
    sign = get_md5(app_id + get_file_md5(image_path) + str(salt) + cuid + mac + app_key)  
  
    payload = {  
        'from': from_lang,  
        'to': to_lang,  
        'appid': app_id,  
        'salt': salt,  
        'sign': sign,  
        'cuid': cuid,  
        'mac': mac,  
    }  
  
    files = {'image': open(image_path, 'rb')} 
    response = requests.post(endpoint, data=payload, files=files)  
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

if __name__ == '__main__':  
    translate_image("")