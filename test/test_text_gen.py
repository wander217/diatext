import json
import random
import cv2
from dataset.text_generator import generator

bg_root = r'D:\adb\asset\bg'
font_root = r'D:\adb\asset\font'
path: str = r'D:\adb\asset\train.txt'
with open(path, 'r', encoding='utf-8') as f:
    data = json.loads(f.readline())
index = random.randint(0, len(data) - 1)
print(index)
bg, words = generator(**data[index], bg_root=bg_root, font_root=font_root)
cv2.imshow("abc", bg)
cv2.waitKey(0)
print(words)
