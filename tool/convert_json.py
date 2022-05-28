import yaml
import json

target = r'D:\workspace\project\diatext\config\dbpp_se_eb0.yaml'
save = r'D:\workspace\project\diatext\config\dbpp_se_eb0.json'
with open(target, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
with open(save, 'w', encoding='utf-8') as f:
    f.write(json.dumps(data))
