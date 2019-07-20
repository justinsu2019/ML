# using baidu API, need self research on how to complete.

from aip import AipOcr
import requests
import json
import base64

APP_ID='16271338'
API_KEY='BbwpnMUW99SVnEHYeEvFRr4F'
SECRET_KEY='DSBN8HYCLipBpc7nTU3kZb8gXMXPTq9o'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

### Input & output file location ###
Filepath = 'C:/Users/guosu/Desktop/1.png'
output_path = 'C:/Users/guosu/Desktop/1.txt' 
### Input & output file location ###


class BaiduImg():
    def __init__(self, img_path):
        self.img_path = Filepath # 传入图片地址

    """ 读取图片 """

    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def img_ocr(self):
        
        image = self.get_file_content(self.img_path)

        """ 调用通用文字识别, 图片参数为本地图片 """
        client.basicGeneral(image)
        """ 如果有可选参数 """
        options = {}
        options["language_type"] = "CHN_ENG"
        options["detect_direction"] = "true"
        options["detect_language"] = "true"
        options["probability"] = "true"
        """ 带参数调用通用文字识别, 图片参数为本地图片 """
        bendi = client.basicGeneral(image, options)
        return bendi  # 返回字典数据
    
    def save_tofile(self, inputs, output_path):
        with open(output_path, 'w+') as fp:
            fp.write(inputs)
            fp.close()


baidu = BaiduImg('xx.png')
rst = baidu.img_ocr()
rst_words = []
i = 0
while i < len(rst['words_result']):
    rst_words.append(rst['words_result'][i]['words'])
    i+=1

b = str()
b += ''.join([str(a) for a in rst_words])

baidu.save_tofile(b ,output_path)
