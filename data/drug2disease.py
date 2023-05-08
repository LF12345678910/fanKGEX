# encoding: utf-8
# @author: fan
# @file: drug2disease.py
# @time: 2023/3/8 10:31


import requests
import random
import csv

# 获取药物列表
def get_drugs():
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/624613/target/1914/substances/txt"
    response = requests.get(url)
    drugs = response.content.decode('utf-8').split('\n')[1:-1] # 去掉标题和最后一行空行
    return drugs

# 获取疾病列表
def get_diseases():
    url = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.owl"
    response = requests.get(url)
    diseases = []
    for line in response.content.decode('utf-8').split('\n'):
        if "<rdfs:label rdf:datatype=" in line:
            diseases.append(line.split('>')[1].split('<')[0])
    return diseases

# 随机生成药物治疗疾病三元组
def generate_triplets(drugs, diseases, num_triplets):
    triplets = []
    for i in range(num_triplets):
        drug = random.choice(drugs)
        disease = random.choice(diseases)
        triplets.append((drug, "treats", disease))
    return triplets

# 将三元组写入CSV文件
def write_to_csv(triplets):
    with open('triplets.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for triplet in triplets:
            writer.writerow(triplet)

# 获取药物和疾病列表
drugs = get_drugs()
diseases = get_diseases()

# 生成一万条随机的药物治疗疾病三元组
num_triplets = 10000
triplets = generate_triplets(drugs, diseases, num_triplets)

# 将三元组写入CSV文件
write_to_csv(triplets)
