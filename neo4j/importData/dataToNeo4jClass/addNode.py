#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/31 14:08
# @Author  : fan

# -*- coding: utf-8 -*-
from py2neo import Node, Graph, Relationship,NodeMatcher
import os
import pandas as pd


class DataToNeo4j(object):
    """将excel中数据存入neo4j"""

    def __init__(self,s_label,o_label):
        """建立连接"""
        # 老版本写法
        # link = Graph("http://localhost:7474", username="neo4j", password="12345678")
        link = Graph("http://localhost:7474/browser/", auth=("neo4j", "12345678"))
        self.graph = link
        # self.graph = NodeMatcher(link)
        # 定义label（定义标签只有这里）
        self.s = s_label
        # self.s = 'Pharmaceutical'
        self.o = o_label
        # self.graph.delete_all()
        self.matcher = NodeMatcher(link)

    def create_node(self, node_s_key, node_o_key):
        """建立节点"""
        for name in node_s_key:
            s_node = Node(self.s, name=name)
            self.graph.merge(s_node,self.s,'name')
        for name in node_o_key:
            o_node = Node(self.o, name=name)
            self.graph.merge(o_node,self.o,'name')


    def create_relation(self, df_data):
        """建立联系"""
        m = 0
        for m in range(0, len(df_data)):
            try:
                print(list(self.matcher.match(self.s).where("_.name=" + "'" + df_data['subject'][m] + "'")))
                print(list(self.matcher.match(self.o).where("_.name=" + "'" + df_data['object'][m] + "'")))
                rel = Relationship(self.matcher.match(self.s).where("_.name=" + "'" + df_data['subject'][m] + "'").first(),
                                   df_data['predication'][m], self.matcher.match(self.o).where("_.name=" + "'" + df_data['object'][m] + "'").first())

                self.graph.merge(rel)
            except AttributeError as e:
                print(e, m)


def data_extraction():
    """节点数据抽取"""

    node_s_key = []
    for i in range(0, len(invoice_data)):
        node_s_key.append(invoice_data['subject'][i])

    node_o_key = []
    for i in range(0, len(invoice_data)):
        node_o_key.append(invoice_data['object'][i])

    # 去除重复的节点名称
    node_s_key = list(set(node_s_key))
    node_o_key = list(set(node_o_key))

    # value抽出作node
    node_list_value = []
    for i in range(0, len(invoice_data)):
        for n in range(1, len(invoice_data.columns)):
            # 取出表头名称invoice_data.columns[i]
            node_list_value.append(invoice_data[invoice_data.columns[n]][i])
    # 去重
    node_list_value = list(set(node_list_value))
    # 将list中浮点及整数类型全部转成string类型
    node_list_value = [str(i) for i in node_list_value]

    return node_s_key, node_o_key, node_list_value


def relation_extraction():
    """联系数据抽取"""

    links_dict = {}
    s_list = []
    p_list = []
    o_list = []

    for i in range(0, len(invoice_data)):
        s_list.append(invoice_data[invoice_data.columns[0]][i])  # 主
        p_list.append(invoice_data[invoice_data.columns[1]][i])  # 关系
        o_list.append(invoice_data[invoice_data.columns[2]][i])  # 谓语

    # 将数据中int类型全部转成string
    s_list = [str(i) for i in s_list]
    p_list = [str(i) for i in p_list]
    o_list = [str(i) for i in o_list]

    # 整合数据，将三个list整合成一个dict
    links_dict['subject'] = s_list
    links_dict['predication'] = p_list
    links_dict['object'] = o_list
    # 将数据转成DataFrame
    df_data = pd.DataFrame(links_dict)
    print("df_data", df_data)
    return df_data


if __name__ == '__main__':
    # 民族药为主实体
    file_path_m = ["../../../data/triples/来源物.xls",
                 "../../../data/triples/化学成分.xls",
                 "../../../data/triples/产地.xls",
                 "../../../data/triples/保存方式.xls",
                 "../../../data/triples/民族药药性.xls",
                 "../../../data/triples/民族药功能.xls",
                 "../../../data/triples/民族药疾病.xls",
                 "../../../data/triples/民族药使用方法.xls",
                 "../../../data/triples/民族药剂型.xls",
                 "../../../data/triples/民族药中药材.xls",
                 "../../../data/triples/民族药药剂.xls"]

    label_m = ['Source','Chemical_composition','Place_of_origin','Preservation_mode','Medicinal_property',
             'Function','Disease','Usage_method','Dosage_form','Chinese_herbal_medicine','Pharmaceutical']

    # 药剂为主实体
    file_path_y = ["../../../data/triples/主体为药剂/药剂药性.xls",
                   "../../../data/triples/主体为药剂/药剂功能.xls",
                   "../../../data/triples/主体为药剂/药剂疾病.xls",
                   "../../../data/triples/主体为药剂/药剂使用方法.xls",
                   "../../../data/triples/主体为药剂/药剂剂型.xls",
                   "../../../data/triples/主体为药剂/药剂中药材.xls"]

    label_y = ['Medicinal_property','Function','Disease','Usage_method','Dosage_form','Chinese_herbal_medicine']

    'Ethnic_medicine'
    'Synonymous_name'
    'Source'
    'Chemical_composition'
    'Place_of_origin'
    'Preservation_mode'
    'Medicinal_property'
    'Function'
    'Disease'
    'Usage_method'
    'Dosage_form'
    'Chinese_herbal_medicine'
    'Pharmaceutical'

    for i in range(len(file_path_m)):
        invoice_data = pd.read_excel(file_path_m[i], header=0)
        o_label = label_m[i]
        s_label = 'Ethnic_medicine'
        relation_extraction()
        create_data = DataToNeo4j(s_label, o_label)
        create_data.create_node(data_extraction()[0], data_extraction()[1])
        create_data.create_relation(relation_extraction())

    print("===================================药剂为主体========================================")
    for i in range(len(file_path_y)):
        invoice_data = pd.read_excel(file_path_y[i], header=0)
        o_label = label_y[i]
        s_label = 'Pharmaceutical'
        relation_extraction()
        create_data = DataToNeo4j(s_label, o_label)
        create_data.create_node(data_extraction()[0], data_extraction()[1])
        create_data.create_relation(relation_extraction())

    print("===导入完成===导入完成===导入完成===导入完成===导入完成===导入完成===导入完成===导入完成===导入完成===导入完成===")




