# -*- coding: utf-8 -*-
"""medical_re.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nddzA4hsk1pr9u1HobaxbAJr2M1QAr-K
"""

import random
import json
import numpy as np
import torch
import torch.nn as nn
# from constant import ProductionConfig as Path
from transformers import BertTokenizer, BertModel, AdamW
from itertools import cycle
import gc
import random
import time
import re
from fanKGEX.model_re.config import config,gpuConfig
from fanKGEX.model_re.model_ex import Model4s,Model4po,loss_fn
import matplotlib.pyplot as plt
from drow import draw_charts
device = gpuConfig.device


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, random):
        super(IterableDataset).__init__()
        self.data = data
        self.random = random
        self.tokenizer = config.tokenizer

    def __len__(self):
        return len(self.data)

    def search(self, sequence, pattern):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    # 预处理数据
    def process_data(self):
        idxs = list(range(len(self.data)))
        if self.random:
            np.random.shuffle(idxs)
        batch_size = config.batch_size
        max_seq_len = config.max_seq_len
        num_p = config.num_p
        # 初始化input id,词的id,三个参数之一
        batch_token_ids = np.zeros((batch_size, max_seq_len), dtype=np.int)
        # 哪些词参与到我们的注意力机制计算当中，这些词的id，三个参数之一
        batch_mask_ids = np.zeros((batch_size, max_seq_len), dtype=np.int)
        # 因为输入是多句话，指定第几句话：0‘58；00，因为我们输入的都是一句话，for了，所以这个参数用不到，都是零
        batch_segment_ids = np.zeros((batch_size, max_seq_len), dtype=np.int)
        # 主体batch_size，乘上个二维，因为一个start位置，一个end位置
        batch_subject_ids = np.zeros((batch_size, 2), dtype=np.int)
        # 1'02'05
        batch_subject_labels = np.zeros((batch_size, max_seq_len, 2), dtype=np.int)
        batch_object_labels = np.zeros((batch_size, max_seq_len, num_p, 2), dtype=np.int)
        batch_i = 0
        for i in idxs:
            text = self.data[i]['text']
            batch_token_ids[batch_i, :] = self.tokenizer.encode(text, max_length=max_seq_len, pad_to_max_length=True,
                                                                add_special_tokens=True)
            batch_mask_ids[batch_i, :len(text) + 2] = 1  # 对pad出来的设置为0，有两个特殊字符
            spo_list = self.data[i]['spo_list']
            idx = np.random.randint(0, len(spo_list), size=1)[0]  # 相当于每次都是随机选一个S来组成数据
            s_rand = self.tokenizer.encode(spo_list[idx][0])[1:-1]  # S的id编码
            s_rand_idx = self.search(list(batch_token_ids[batch_i, :]), s_rand)
            batch_subject_ids[batch_i, :] = [s_rand_idx, s_rand_idx + len(s_rand) - 1]
            for i in range(len(spo_list)):
                spo = spo_list[i]
                s = self.tokenizer.encode(spo[0])[1:-1]
                p = config.prediction2id[spo[1]]
                o = self.tokenizer.encode(spo[2])[1:-1]
                s_idx = self.search(list(batch_token_ids[batch_i]), s)
                o_idx = self.search(list(batch_token_ids[batch_i]), o)
                if s_idx != -1 and o_idx != -1:
                    batch_subject_labels[batch_i, s_idx, 0] = 1
                    batch_subject_labels[batch_i, s_idx + len(s) - 1, 1] = 1
                    if s_idx == s_rand_idx:
                        batch_object_labels[batch_i, o_idx, p, 0] = 1
                        batch_object_labels[batch_i, o_idx + len(o) - 1, p, 1] = 1
            batch_i += 1
            if batch_i == batch_size or i == idxs[-1]:
                yield batch_token_ids, batch_mask_ids, batch_segment_ids, batch_subject_labels, batch_subject_ids, batch_object_labels
                batch_token_ids[:, :] = 0
                batch_mask_ids[:, :] = 0
                batch_subject_ids[:, :] = 0
                batch_subject_labels[:, :, :] = 0
                batch_object_labels[:, :, :, :] = 0
                batch_i = 0

    def get_stream(self):
        return cycle(self.process_data())

    def __iter__(self):
        return self.get_stream()


def train(train_data_loader, model4s, model4po, optimizer):
    losses = []
    for epoch in range(config.EPOCH):
        begin_time = time.time()
        model4s.train()
        model4po.train()
        train_loss = 0.
        for bi, batch in enumerate(train_data_loader):
            if bi >= len(train_data_loader) // config.batch_size:
                break
            # print("batch", batch)
            batch_token_ids, batch_mask_ids, batch_segment_ids, batch_subject_labels, batch_subject_ids, batch_object_labels = batch
            # print("batch_object_labels维度",batch_object_labels,batch_object_labels.shape)
            batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(device)
            batch_mask_ids = torch.tensor(batch_mask_ids, dtype=torch.long).to(device)
            batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long).to(device)
            batch_subject_labels = torch.tensor(batch_subject_labels, dtype=torch.float).to(device)
            batch_object_labels = torch.tensor(batch_object_labels, dtype=torch.float).view(config.batch_size,
                                                                                            config.max_seq_len,
                                                                                            config.num_p * 2).to(device)
            # print("batch_object_labels维度2", batch_object_labels.shape)
            batch_subject_ids = torch.tensor(batch_subject_ids, dtype=torch.int)

            #
            batch_subject_labels_pred, hidden_states = model4s(batch_token_ids, batch_mask_ids, batch_segment_ids)
            # 计算了预测标签与真实标签之间的损失值
            loss4s = loss_fn(batch_subject_labels_pred, batch_subject_labels.to(torch.float32))
            # 计算了损失值的平均值。dim=2 表示对第二个维度（类别数或标签数）进行平均，将平均值乘以掩码
            loss4s = torch.mean(loss4s, dim=2, keepdim=False) * batch_mask_ids
            # 对所有样本的平均损失值进行求和
            loss4s = torch.sum(loss4s)
            # 对损失值进行加权平均
            loss4s = loss4s / torch.sum(batch_mask_ids)

            batch_object_labels_pred = model4po(hidden_states, batch_subject_ids, batch_mask_ids)
            loss4po = loss_fn(batch_object_labels_pred, batch_object_labels.to(torch.float32))
            loss4po = torch.mean(loss4po, dim=2, keepdim=False) * batch_mask_ids
            loss4po = torch.sum(loss4po)
            loss4po = loss4po / torch.sum(batch_mask_ids)

            loss = loss4s + loss4po
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            losses.append(float(loss.item()))
            print('batch:', bi, 'loss:', float(loss.item()))

        print('final train_loss:', train_loss / len(train_data_loader) * config.batch_size, 'cost time:',
              time.time() - begin_time)

        # 绘制损失函数曲线图
    plt.plot(losses)
    plt.title('Loss Curve')
    # plt.xlim([0, config.EPOCH])    # 自定义横坐标范围
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    del train_data_loader
    gc.collect()

    return {
        "model4s_state_dict": model4s.state_dict(),  # model.state_dict() 是 PyTorch 中的一个方法，用于获取模型的参数字典
        "model4po_state_dict": model4po.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def run_train():
    load_schema(config.PATH_SCHEMA)
    train_path = config.PATH_TRAIN
    all_data = load_data(train_path)
    random.shuffle(all_data)    # 打乱数据顺序

    # 8:2划分训练集、验证集
    idx = int(len(all_data) * 0.8)
    train_data = all_data[:idx]
    valid_data = all_data[idx:]

    # train
    train_data_loader = IterableDataset(train_data, True)
    num_train_data = len(train_data)
    # print(num_train_data)

    model4s = Model4s()
    model4s.to(device)   # 将模型转移到gpu上计算

    model4po = Model4po()
    model4po.to(device)

    # 加载两个预训练模型，新建优化器对象，指定optimizer的参数，比如学习率，权重衰减等。还可以指定哪些参数不参与优化
    # https://www.jianshu.com/p/c6cce168f3e3
    param_optimizer = list(model4s.named_parameters()) + list(model4po.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 模型的权重，和学习率衰减的相关参数，可调
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 初始学习率
    lr = config.learning_rate
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    checkpoint = train(train_data_loader, model4s, model4po, optimizer)

    del train_data
    gc.collect()
    # save
    model_path = config.PATH_SAVE
    torch.save(checkpoint, model_path)
    print('saved!')

    # valid
    model4s.eval()
    model4po.eval()
    f1, precision, recall = evaluate(valid_data, True, model4s, model4po)
    print('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))


# Schema是一种数据结构，用于表示一组实体和它们之间的关系。
def load_schema(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
        predicate = list(data.keys())
        prediction2id = {}
        id2predicate = {}
        for i in range(len(predicate)):
            prediction2id[predicate[i]] = i
            id2predicate[i] = predicate[i]
    num_p = len(predicate)
    config.prediction2id = prediction2id
    config.id2predicate = id2predicate
    config.num_p = num_p


def load_data(path):
    text_spos = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
        for item in data:
            text = item['text']
            spo_list = item['spo_list']
            text_spos.append({
                'text': text,
                'spo_list': spo_list
            })
    return text_spos


# 提取spo三元组
def extract_spoes(text, model4s, model4po):
    """
    return: a list of many tuple of (s, p, o)
    """
    # 处理text
    with torch.no_grad():
        tokenizer = config.tokenizer
        max_seq_len = config.max_seq_len
        token_ids = torch.tensor(
            tokenizer.encode(text, max_length=max_seq_len, pad_to_max_length=True, add_special_tokens=True)).view(1, -1).to(device)
        if len(text) > max_seq_len - 2:
            text = text[:max_seq_len - 2]
        mask_ids = torch.tensor([1] * (len(text) + 2) + [0] * (max_seq_len - len(text) - 2)).view(1, -1).to(device)
        segment_ids = torch.tensor([0] * max_seq_len).view(1, -1).to(device)

        subject_labels_pred, hidden_states = model4s(token_ids, mask_ids, segment_ids)
        subject_labels_pred = subject_labels_pred.cpu()
        subject_labels_pred[0, len(text) + 2:, :] = 0
        start = np.where(subject_labels_pred[0, :, 0] > 0.4)[0]
        end = np.where(subject_labels_pred[0, :, 1] > 0.4)[0]

        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))

        if len(subjects) == 0:
            return []
        subject_ids = torch.tensor(subjects).view(1, -1)

        spoes = []
        for s in subjects:
            object_labels_pred = model4po(hidden_states, subject_ids, mask_ids)
            object_labels_pred = object_labels_pred.view((1, max_seq_len, config.num_p, 2)).cpu()
            object_labels_pred[0, len(text) + 2:, :, :] = 0
            start = np.where(object_labels_pred[0, :, :, 0] > 0.4)
            end = np.where(object_labels_pred[0, :, :, 1] > 0.4)

            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append((s, predicate1, (_start, _end)))
                        break

    id_str = ['[CLS]']
    i = 1
    index = 0
    while i < token_ids.shape[1]:
        if token_ids[0][i] == 102:
            break

        word = tokenizer.decode(token_ids[0, i:i + 1])
        word = re.sub('#+', '', word)
        if word != '[UNK]':
            id_str.append(word)
            index += len(word)
            i += 1
        else:
            j = i + 1
            while j < token_ids.shape[1]:
                if token_ids[0][j] == 102:
                    break
                word_j = tokenizer.decode(token_ids[0, j:j + 1])
                if word_j != '[UNK]':
                    break
                j += 1
            if token_ids[0][j] == 102 or j == token_ids.shape[1]:
                while i < j - 1:
                    id_str.append('')
                    i += 1
                id_str.append(text[index:])
                i += 1
                break
            else:
                index_end = text[index:].find(word_j)
                word = text[index:index + index_end]
                id_str.append(word)
                index += index_end
                i += 1
    res = []
    for s, p, o in spoes:
        s_start = s[0]
        s_end = s[1]
        sub = ''.join(id_str[s_start:s_end + 1])
        o_start = o[0]
        o_end = o[1]
        obj = ''.join(id_str[o_start:o_end + 1])
        res.append((sub, config.id2predicate[p], obj))

    return res


class SPO(tuple):
    def __init__(self, spo):
        self.spox = (
            tuple(config.tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(config.tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


# 评估函数
def evaluate(data, is_print, model4s, model4po):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'], model4s, model4po)])  # 模型提取出的三元组数目
        T = set([SPO(spo) for spo in d['spo_list']])  # 正确的三元组数目
        if is_print:
            print('text:', d['text'])
            print('R:', R)
            print('T:', T)
        X += len(R & T)  # 模型提取出的三元组数目中正确的个数
        Y += len(R)  # 模型提取出的三元组个数
        Z += len(T)  # 正确的三元组总数
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    return f1, precision, recall


def load_model():
    load_schema(config.PATH_SCHEMA)
    checkpoint = torch.load(config.PATH_MODEL, map_location='cpu')

    model4s = Model4s()
    model4s.load_state_dict(checkpoint['model4s_state_dict'],strict=False)
    # model4s.cuda()

    model4po = Model4po()
    model4po.load_state_dict(checkpoint['model4po_state_dict'],strict=False)
    # model4po.cuda()

    return model4s, model4po


def get_triples(content, model4s, model4po):
    if len(content) == 0:
        return []
    text_list = content.split('。')[:-1]
    res = []
    for text in text_list:
        if len(text) > 128:
            text = text[:128]
        triples = extract_spoes(text, model4s, model4po)
        res.append({
            'text': text,
            'triples': triples
        })
    return res


if __name__ == "__main__":
    # pre_model_path = r'D:\A_TheStoragePool\MyDirection\KnowledgeGraph\MyCode\KgExtract\fanKGEX\model\re\medical_re\pytorch_model.bin'
    # checkpoint = torch.load(pre_model_path)
    # # print(checkpoint.state_dict())
    # keys = list(checkpoint.keys())
    # print(keys)

    model2s = Model4s()
    print(model2s)

    # with open(config.PATH_TRAIN, 'r', encoding="utf-8", errors='replace') as f:
    #     data = json.load(f)
    #
    #     f1=open("train.json","w")
    #
    #     json.dump(data,f1,ensure_ascii=False,indent=True)
    #     print("finish")

    # load_schema(config.PATH_SCHEMA)
    # model4s, model4po = load_model()
    #
    # text = "据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、=乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。"
    #
    # res = get_triples(text, model4s, model4po)

    # print(res)

