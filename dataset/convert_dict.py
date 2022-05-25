import json
import random

dict_path: str = r'D:\adb\asset\dict.txt'
save_path: str = r'D:\adb\asset\new_dict.txt'
data = []
special = ['!', '@', '#', '$', '%',
           '^', '&', '*', '(', ')', '-',
           '+', '=', '[', ']', '{', '}',
           ':', ';', '"', "'", ',', '.',
           '/', '?', '<', '>']
with open(dict_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().strip("\r\t").strip("\n")
        data.append(tmp)
        data.append(tmp + special[random.randint(0, len(special) - 1)])
        data.append(special[random.randint(0, len(special) - 1)] + tmp)
    for i in range(0, 1000):
        data.append(str(i % 10))
    for i in range(0, 1000):
        data.append("0" + str(random.randint(10, 10000000)))

with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data))
