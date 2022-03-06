import requests
import json


texts = ["患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术(dixon术),手术过程顺利。", 
          "患者1个月前无明显诱因出现下腹部不适,进食硬质食物时有哽噎感,哽咽感常在吞咽水后缓解消失,进食流质无明显不适。当时无发热、呕吐,无乏力,无声音嘶哑,无呕血黑便,无头晕头痛,无腹泻腹胀,无黄疸"] 

data = {"text_list": texts}

r = requests.post(url="http://127.0.0.1:8888/service/api/medical_ner", json=data)

res_data = r.json()

print(json.dumps(res_data, ensure_ascii=False, indent=4))