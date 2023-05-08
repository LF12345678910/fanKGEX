# encoding: utf-8
# @author: fan
# @file: outputSave.py.py
# @time: 2023/3/2 11:26


from model_re import medical_re
from model_re import config
import json
import csv

# medical_re.load_schema()
model4s, model4po = medical_re.load_model()

# text = '据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括:1.自身免疫系统缺陷\n2.人传人。'
# 要读成字符串
file = open(r"D:\A_TheStoragePool\MyDirection\KnowledgeGraph\MyCode\KgExtract\fanKGEX\data\txt\test.txt", 'r', encoding='utf-8')
text = file.read()
# print(text)
res = medical_re.get_triples(text, model4s, model4po)

# print(json.dumps(res, ensure_ascii=False, indent=True))
r = []
for i in range(len(res)):
    r1 = res[i]['triples']
    for j in range(len(r1)):
        r.append(r1[j])

print(r,type(r))
# print(res,type(res))


# 将元组列表写入csv
# data=[('name', 'sex', 'date'), ('x1', 'f', '1948/05/28'), ('x2', 'm', '1952/03/27'), ('x3', 'f', '1994/12/09'), ('x4', 'f', '1969/08/02')]
data = r
# 不加encoding='utf8'的话，默认是gbk格式
with open(r"D:\A_TheStoragePool\MyDirection\KnowledgeGraph\MyCode\KgExtract\fanKGEX\data\csv\test2kongge.csv","w+",encoding='utf8') as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\r\n")
    # writer = csv.writer(f, delimiter=" ", lineterminator="\r\n")
    writer.writerows(data)