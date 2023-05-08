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
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        # 定义一个线性层，用于标签预测
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_tags,bias=True)
        self.sigmoid = nn.Sigmoid()
        # 定义一个 CRF 层，用于联合抽取
        self.crf = CRF(num_tags)

    # input_ids（batch_token_ids）：输入序列的token ID。它是一个形状为（batch_size，seq_length）的张量
    # attention_mask：注意力掩码，用于指示哪些标记应被包括在注意力计算中，哪些应该被忽略。它是形状为（batch_size，seq_length）的张量
    # token_type_ids（batch_segment_ids）：用于区分不同句子的标记ID。它是形状为（batch_size，seq_length）的张量
    def forward(self, input_ids, input_mask, segment_ids):
        # 使用 BERT 模型进行文本编码
        # # 取出 BERT 的所有输出序列[0]，其中第一个是最后一层（bert输出层)的隐藏状态
        bert_output = self.bert(input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids)[0]  # (batch_size, sequence_length, hidden_size)
        # inputs = bert_output.clone().detach()
        hidden_states, _ = self.lstm(bert_output)
        hidden_states = hidden_states.clone().detach()  # 使用 clone().detach() 快捷地创建了一个新的 Tensor，代替了原始的 Tensor
        # hidden_states = hidden_states[:, -1, :]  # 取最后一个时间步的状态
        # pow(x,y)：表示x的y次幂。
        output = self.sigmoid(
            self.linear(
                self.dropout(hidden_states)
            )).pow(2)  # output是Tensor格式，二维数组，这里是把数组里的每个元素取平方

        # 加crf层
        # logits = output1.permute(1, 0, 2)  # 维度转换，将时间步作为第一维
        # mask = (attention_mask == 1).byte()  # 生成一个掩码，用于在 CRF 层中过滤无效的预测
        # mask = mask.T
        # tags = self.crf.decode(logits,mask=mask)  # 使用 CRF 层对标签概率分布进行联合抽取，得到最终的标签序列
        return output,hidden_states  # 返回所有时刻的隐藏状态以及最终的标签序列

# class Model4po(nn.Module):
#     def __init__(self, hidden_size=768, num_tags=16, input_size=768):
#         super(Model4po, self).__init__()
#         self.bert = BertModel.from_pretrained(config.PATH_BERT)
#         for param in self.bert.parameters():  # 对bert层的参数做初始化
#             param.requires_grad = True   # 是否进行Finetune
#         self.lstm = nn.LSTM(input_size=input_size,
#                             # input_size=hidden_size,
#                             hidden_size=hidden_size // 2,
#                             num_layers=1,
#                             batch_first=True,
#                             bidirectional=True)
#         self.linear = nn.Linear(in_features=hidden_size, out_features=num_tags * 2)
#         # self.crf = CRF(num_tags)
#         self.crf = CRF(num_tags * 2)
#
#     def forward(self, input_ids, attention_mask, token_type_ids, hidden_states):
#         sequence_output = hidden_states  # 使用来自第一个模型的隐藏状态作为输入序列
#         lstm_output, _ = self.lstm(sequence_output)  # 将输入序列传入 LSTM，输出的结果为所有时刻的隐藏状态
#         # 使用 BERT 模型对文本进行编码
#         outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         bert_output = outputs[0]
#         # 对lstm_output进行pca降维，因为后面维度不符
#         # lstm_output_np = lstm_output.detach().cpu().numpy()
#         # pca = PCA(n_components=768)
#         # lstm_output_pca_np = pca.fit_transform(lstm_output_np)
#         # lstm_output_pca = torch.from_numpy(lstm_output_pca_np)
#         # 将 LSTM 输出和 BERT 输出进行拼接，并使用线性层对拼接后的输出进行标签预测，得到一个标签概率分布
#         output = self.linear(lstm_output + bert_output)
#         # output = torch.cat([lstm_output, bert_output], dim=1)
#         # output = self.linear(torch.cat([lstm_output, bert_output], dim=0))  # 修改合并行为的维度
#
#         # crf层
#         logits = output.permute(1, 0, 2)  # 维度转换，将时间步作为第一维
#         mask = (attention_mask == 1).byte()  # 生成一个掩码，用于在 CRF 层中过滤无效的预测
#         mask = mask.T
#         tags = self.crf.decode(logits,mask=mask)  # 使用 CRF 层对标签概率分布进行联合抽取，得到最终的标签序列
#         return output  # 主要是用来外面计算损失的


class Model4po(nn.Module):
    def __init__(self, num_p=config.num_p, hidden_size=768):
        super(Model4po, self).__init__()
        self.dropout = nn.Dropout(p=0.4)  # 在训练期间，使用伯努利分布的样本，以概率p随机归零输入张量的一些元素。
        # 通过linear将1×512×768维的隐藏层，降成1×512×16的参数（都是tensor），因为本质是16分类
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_p * 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, batch_subject_ids, input_mask):
        all_s = torch.zeros((hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]),
                            dtype=torch.float32)

        for b in range(hidden_states.shape[0]):
            s_start = batch_subject_ids[b][0]
            s_end = batch_subject_ids[b][1]
            s = hidden_states[b][s_start] + hidden_states[b][s_end]
            cue_len = torch.sum(input_mask[b])
            all_s[b, :cue_len, :] = s
        hidden_states += all_s.to(device)

        output = self.sigmoid(self.linear(self.dropout(hidden_states))).pow(4)

        return output  # (batch_size, max_seq_len, num_p*2)


# 损失函数
def loss_fn(pred, target):
    # 二元交叉熵损失函数，多用于多分类
    loss_fct = nn.BCELoss(reduction='none').to(device)
    # loss_fct = nn.BCEWithLogitsLoss(reduction='none').to(device)
    # 交叉熵损失函数
    # loss_fct = nn.CrossEntropyLoss(reduction='none').to(device)
    # 均方误差（MSE）损失函数，用于回归
    # loss_fct = nn.MSELoss(reduction='none').to(device)
    return loss_fct(pred, target)


if __name__ == '__main__':
     model = Model4s()
     print(model)