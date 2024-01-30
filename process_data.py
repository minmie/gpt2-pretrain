import json
import pandas as pd




f = open('./all_train.json', 'w', encoding='utf8')


with open('data.jsonl', 'r', encoding='utf8') as f2:

    for line in f2:
        line = json.loads(line)
        f.write(json.dumps({'content': line['story']}, ensure_ascii=False)+'\n')

data = pd.read_excel('SmoothNLP36kr新闻数据集10k.xlsx')
data = data.dropna(subset=['content'])

for each in data['content'].to_list():
    f.write(json.dumps({'content': each}, ensure_ascii=False)+'\n')

data = pd.read_excel('SmoothNLP金融新闻数据集样本20k.xlsx')
data = data.dropna(subset=['content'])

for each in data['content'].to_list():
    f.write(json.dumps({'content': each}, ensure_ascii=False)+'\n')

with open("/home/chenjq/datasets/LCSTS_new/train.json", 'r', encoding='utf8') as f2:

    for line in f2:
        line = json.loads(line)
        f.write(json.dumps({'content': line['content']}, ensure_ascii=False)+'\n')

with open('./wikipedia-cn-20230720-filtered.json', 'r', encoding='utf8') as f2:
    datas = json.load(f2)
    for each in datas:
        f.write(json.dumps({'content': each['completion']}, ensure_ascii=False) + '\n')

f.close()