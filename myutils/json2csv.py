#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 9:07
# @Author  : fan
import csv
import json


datacsv = []
# 读取json文件内容,返回字典格式
with open('../data/json/weiyaoOut.json','r',encoding='utf8')as fp:
    data = fp.read()
    json_data = json.loads(data)

    for item in json_data:
        # print(item["spo_list"])
        for triples in item["spo_list"]:
            datacsv.append(triples)

    # print(datacsv)

data = datacsv
# 不加encoding='utf8'的话，默认是gbk格式
with open(r"D:\A_TheStoragePool\MyDirection\KnowledgeGraph\MyCode\KgExtract\fanKGEX\data\csv\weiyaoTriple.csv","w+",encoding='utf8') as f:
    # writer = csv.writer(f, delimiter="\t", lineterminator="\r\n")
    writer = csv.writer(f, delimiter=" ", lineterminator="\r\n")
    writer.writerows(data)