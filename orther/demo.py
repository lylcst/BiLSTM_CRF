import json


tag2idx = {"<START_TAG>": 0, "<END_TAG>": 1}
token2idx = {"<PAD>": 0, "<UNK>": 1}

data_lines = []
with open("../data/train.txt", "rt", encoding="utf-8") as f:
    lines = f.readlines()
    data_lines.extend(lines)

with open("../data/test.txt", "rt", encoding="utf-8") as f:
    lines = f.readlines()
    data_lines.extend(lines)

with open("../data/dev.txt", "rt", encoding="utf-8") as f:
    lines = f.readlines()
    data_lines.extend(lines)


for line in data_lines:
    if not line.strip():
        continue
    token, tag = line.strip().split()
    if token not in token2idx:
        token2idx[token] = len(token2idx)
    if tag not in tag2idx:
        tag2idx[tag] = len(tag2idx)

print(tag2idx)
print(token2idx)

with open("../data/tag2idx.json", "wt", encoding="utf-8") as f:
    json.dump(tag2idx, f, ensure_ascii=False, indent=4)

with open("../data/token2idx.json", "wt", encoding="utf-8") as f:
    json.dump(token2idx, f, ensure_ascii=False, indent=4)
        
