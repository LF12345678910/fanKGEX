#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 11:24
# @Author  : fan

from transformers import BertTokenizer, BertModel, AdamW
from fanKGEX.model_re.config import config,gpuConfig
import torch
from torchcrf import CRF
# from torchcrf import
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

device = gpuConfig.device


class Model4s(nn.Module):
    def __init__(self, hidden_size=768, num_tags=2):   # num_tags主实体2个
        super(Model4s, self).__init__()
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(config.PATH_BERT)
        for param in self.bert.parameters():  # 对bert层的参数做初始化
            param.requires_grad = True   # 是否进行Finetune
        # 定义一个双向 LSTM 层
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        # 定义一个线性层，用于标签预测
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_tags)
        # 定义一个 CRF 层，用于联合抽取
        self.crf = CRF(num_tags)

    # input_ids（batch_token_ids）：输入序列的token ID。它是一个形状为（batch_size，seq_length）的张量
    # attention_mask：注意力掩码，用于指示哪些标记应被包括在注意力计算中，哪些应该被忽略。它是形状为（batch_size，seq_length）的张量
    # token_type_ids（batch_segment_ids）：用于区分不同句子的标记ID。它是形状为（batch_size，seq_length）的张量
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用 BERT 模型进行文本编码
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]   # 取出 BERT 的所有输出序列，其中第一个是最后一层（bert输出层)的隐藏状态
        hidden_states, _ = self.lstm(sequence_output)  # 将 BERT 输出序列传入 LSTM，输出的结果为所有时刻的隐藏状态
        output = self.linear(hidden_states)  # 使用线性层对每个时间步的隐藏状态进行标签预测，得到一个标签概率分布


        # 加crf层
        logits = output.permute(1, 0, 2)  # 维度转换，将时间步作为第一维
        mask = (attention_mask == 1).byte()  # 生成一个掩码，用于在 CRF 层中过滤无效的预测
        mask = mask.T
        tags = self.crf.decode(logits,mask=mask)  # 使用 CRF 层对标签概率分布进行联合抽取，得到最终的标签序列
        return tags,hidden_states  # 返回所有时刻的隐藏状态以及最终的标签序列


class Model4po(nn.Module):
    def __init__(self, hidden_size=768, num_tags=16, input_size=768):
        super(Model4po, self).__init__()
        self.bert = BertModel.from_pretrained(config.PATH_BERT)
        for param in self.bert.parameters():  # 对bert层的参数做初始化
            param.requires_grad = True   # 是否进行Finetune
        self.lstm = nn.LSTM(input_size=input_size,
                            # input_size=hidden_size,
                            hidden_size=hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_tags * 2)
        # self.crf = CRF(num_tags)
        self.crf = CRF(num_tags * 2)

    def forward(self, input_ids, attention_mask, token_type_ids, hidden_states):
        sequence_output = hidden_states  # 使用来自第一个模型的隐藏状态作为输入序列
        lstm_output, _ = self.lstm(sequence_output)  # 将输入序列传入 LSTM，输出的结果为所有时刻的隐藏状态
        # 使用 BERT 模型对文本进行编码
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_output = outputs[0]
        # 对lstm_output进行pca降维，因为后面维度不符
        # lstm_output_np = lstm_output.detach().cpu().numpy()
        # pca = PCA(n_components=768)
        # lstm_output_pca_np = pca.fit_transform(lstm_output_np)
        # lstm_output_pca = torch.from_numpy(lstm_output_pca_np)
        # 将 LSTM 输出和 BERT 输出进行拼接，并使用线性层对拼接后的输出进行标签预测，得到一个标签概率分布
        output = self.linear(lstm_output + bert_output)
        # output = torch.cat([lstm_output, bert_output], dim=1)
        # output = self.linear(torch.cat([lstm_output, bert_output], dim=0))  # 修改合并行为的维度

        # crf层
        logits = output.permute(1, 0, 2)  # 维度转换，将时间步作为第一维
        mask = (attention_mask == 1).byte()  # 生成一个掩码，用于在 CRF 层中过滤无效的预测
        mask = mask.T
        tags = self.crf.decode(logits,mask=mask)  # 使用 CRF 层对标签概率分布进行联合抽取，得到最终的标签序列
        return tags  # 主要是用来外面计算损失的


# 模型、输入文本的 ID 序列、输入文本的 attention mask 序列、输入文本的 token type ID 序列、来自第一个模型的隐藏状态序列和真实标签序列。
def loss_fn(model4s, model4po, attention_mask, tags_sub_pred, tags_obj_pred, tags_sub_true, tags_obj_true,batch_subject_ids):
    # 将模型设为训练模式，以便计算损失
    # 将所有输入序列和标签转换为 Tensor 类型，并将它们发送到计算设备上
    tags_sub_true = torch.tensor(tags_sub_true).to(device)
    tags_sub_pred = torch.tensor(tags_sub_pred).to(device)
    tags_obj_true = torch.tensor(tags_obj_true).to(device)
    tags_obj_pred = torch.tensor(tags_obj_pred).to(device)
    batch_subject_ids = torch.tensor(batch_subject_ids).to(device)
    # logits = model4s.linear(hidden_states)  # 使用线性层对每个时间步的隐藏状态进行标签预测，得到一个标签概率分布
    # logits = logits.permute(1, 0, 2)  # 维度转换，将时间步作为第一维
    # 两个tags都是二维，计算crf损失需要三维
    # 把二维张量变为了(batch_size, sequence_length, 1)的三维张量
    # 最后一维期望的是标签的数量，不能小于2
    tags_sub_pred = torch.unsqueeze(tags_sub_pred, dim=2)
    tags_sub_pred = torch.cat([tags_sub_pred, 1 - tags_sub_pred], dim=2)  # 将第三维的大小从1改为2
    # 真实标签不需要变
    # tags = torch.unsqueeze(tags, dim=2)
    # tags1 = torch.cat([tags, 1 - tags], dim=2)  # 将第三维的大小从1改为2
    # 计算 CRF 层的损失函数，得到标签序列的负对数似然损失
    # .crf是由 torchcrf 库提供的一个层，用于计算评估和损失函数
    # tags_pred1: 是模型预测的每个标记的分数矩阵，形状为(batch_size, seq_len, num_tags)，它按时间步的顺序排列。
    # tags: 是相应batch的正确标签序列，形状为(batch_size, seq_len)。它在batch中排列，它保存的是对于每个时间步模型预测的标记的正确答案，即正确的标记序列。
    mask = (attention_mask == 1).byte()  # 生成一个掩码，用于在 CRF 层中过滤无效的预测
    mask = mask[:, :tags_sub_pred.shape[1]]  # 将掩码维度与预测序列 tags_sub_pred 的维度相匹配
    loss_s = -model4s.crf(tags_sub_pred, tags_sub_true, mask = mask)
    # loss_s.requires_grad_()
    # loss_s = 0
    # 计算第二个模型的输出，获取最终的标签序列
    # 将预测序列和真实序列进行对比，生成一个掩码，用于在交叉熵损失函数中过滤掉无效的预测（即 PAD 标记）
    # mask = (attention_mask == 1).float()
    # 标签张量(tags)是形状为(1, 25)的张量
    # 首先将其转换为相应的 one-hot 编码形式 (1, 25, num_classes)
    # loss_po = categorical_crossentropy(tags_obj_true.cpu(),tags_obj_pred.cpu())
    # print(loss_po)
    # exit()



    # tags_obj_pred = tags_obj_pred.transpose(0, 1).float()
    tags_obj_pred = tags_obj_pred.float()
    tags_obj_true = torch.argmax(tags_obj_true,dim=1)


    # print('tags_obj_pred',tags_obj_pred)
    # print('tags_obj_true',tags_obj_true,tags_obj_true.type())
    # exit()
    # 计算交叉熵损失函数，得到标签序列的损失
    # 使用掩码对损失进行加权平均，使得无效的预测不会对总损失产生影响
    # reduction参数用于指定如何对损失进行缩减，取值可以是'mean'，'sum'，'none'之一。
    # avg_weight参数是一个可选的布尔值，当weight参数被指定时用于控制权重的缩放方式。
    # 由于客体和关系可能会使用填充标记，因此需要在损失函数计算时指定为reduce_mean模式
    loss_fn_po = nn.BCELoss(reduction='none')
    # loss_po = loss_fn_po(tags_obj_pred, tags_obj_true)
    loss_po = F.cross_entropy(tags_obj_pred, tags_obj_true, reduction='none')
    loss_po.requires_grad_()
    # print(loss_po)

    loss = loss_s + loss_po
    loss.requires_grad_()
    return loss


import numpy as np

def categorical_crossentropy(y_true, y_pred):
    # 将真实值和预测值都转化为概率分布的形式，例如使用softmax函数
    # 这里假设y_true和y_pred都是(batch_size, num_classes)的矩阵
    y_true = np.exp(y_true) / np.sum(np.exp(y_true), axis=1, keepdims=True)
    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)

    # 计算交叉熵损失
    loss = -np.sum(y_true * np.log(y_pred), axis=1)

    # 返回平均损失
    return np.mean(loss)


# # 初始化两个模型
# model4s = Model4s()
# model4po = Model4po()
#
# # 定义CRF损失函数
# crf_loss = CRF.loss()
#
#
# # 定义总损失函数
# def loss_fn(input_ids, attention_mask, token_type_ids, tags, hidden_states):
#     # 计算模型4s的损失函数
#     s_tags, s_lstm_outputs = model4s(input_ids, attention_mask, token_type_ids)
#     s_loss = crf_loss(s_lstm_outputs.permute(1, 0, 2), tags, mask=attention_mask.byte(), reduction='token_mean')
#
#     # 计算模型4po的损失函数
#     po_tags = model4po(input_ids, attention_mask, token_type_ids, hidden_states)
#     po_loss = crf_loss(model4po.linear.weight.view(1, -1, model4po.num_tags),
#                        po_tags.permute(1, 0),
#                        mask=attention_mask.byte(),
#                        reduction='token_mean')
#
#     # 返回两个模型的损失函数之和
#     return s_loss + po_loss


if __name__ == '__main__':
     model = Model4s()
     print(model)