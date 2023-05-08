# encoding: utf-8
# @author: fan
# @file: jsonl2json.py
# @time: 2023/3/7 14:50

import json
import numpy as np

# 读取json文件内容,返回字典格式
with open('../data/json/doccano.jsonl','r',encoding='utf8')as fp:
    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    papers = []
    for line in fp.readlines():
        json_data = json.loads(line)  # loads() 传的是json字符串，而 load() 传的是文件对象
        sanyuan1 = []
        b = []
        entitie = json_data['entities']
        relation = json_data['relations']
        text = json_data['text']
        # print(relation[0][])
        # exit()
        # print(entitie)
        # print(relation)
        for i in range(len(relation)):
            sanyuan1.append([relation[i]['from_id'],relation[i]['type'],relation[i]['to_id']])
        # print(sanyuan1)

        for j in range(len(sanyuan1)):
            h = ''
            r = ''
            t = ''
            a = []
            for i in range(len(entitie)):
                if entitie[i]['id'] == sanyuan1[j][0]:
                    # print(text[entitie[i]['start_offset']: entitie[i]['end_offset']])
                    h = text[entitie[i]['start_offset']: entitie[i]['end_offset']]
                    r = sanyuan1[j][1]

                if entitie[i]['id'] == sanyuan1[j][2]:
                    # print(text[entitie[i]['start_offset']: entitie[i]['end_offset']])
                    t = text[entitie[i]['start_offset']: entitie[i]['end_offset']]
            a.append([h,r,t])
            b.append(a[0])

        # print("b",b)
        # exit()
        emptyDict = {'text': text, 'spo_list': []}
        for i in range(len(b)):
            emptyDict['spo_list'].append(b[i])
        papers.append(emptyDict)


# print("papers", papers)

# 将字典数据写入到json文件中
list = papers
with open('../data/json/out.json','w',encoding='utf8')as fp:
    json.dump(list,fp,ensure_ascii=False)    # 如果ensure_ascii为false，则返回值可以包含非ascii值