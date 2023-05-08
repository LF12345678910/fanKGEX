#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/5 17:19
# @Author  : fan

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 绘制统计图表
def draw_charts(lassoCV_x, coef, lassoCV_model):
    # 绘制特征相关系数热力图
    f, ax = plt.subplots(figsize=(6, 6))
    # 绘制混淆矩阵
    sns.heatmap(lassoCV_x.corr(), annot=True, cmap='coolwarm', annot_kws={'size': 10, 'weight': 'bold', }, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, va='top', ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
    plt.show()

    # 绘制特征系数的柱状图
    weight = coef[coef != 0].to_dict()
    # 根据值的大小排列一下
    weight = dict(sorted(weight.items(), key=lambda x: x[1], reverse=False))
    plt.figure(figsize=(8, 6))
    plt.title('characters classification weight', fontsize=8)
    plt.xlabel(u'weighted value', fontsize=8)  # 设置x轴，并设定字号大小
    plt.ylabel(u'feature', fontsize=8)
    # plt.barh()：横向的柱状图，可以理解为正常柱状图旋转了90°
    plt.barh(range(len(weight.values())), list(weight.values()), tick_label=list(weight.keys()), alpha=0.6,
             facecolor='blue', edgecolor='black', label='feature weight')
    plt.legend(loc=4)  # 图例展示位置，数字代表第几象限
    plt.show()

    # 绘制误差棒图
    MSEs = lassoCV_model.mse_path_
    mse = []
    std = []
    for m in MSEs:
        mse.append(np.mean(m))
        std.append(np.std(m))
    plt.figure(figsize=(8, 6))
    plt.errorbar(lassoCV_model.alphas_, mse, std, fmt='o', ecolor='lightblue', elinewidth=3, ms=5, mfc='wheat',
                 mec='salmon', capsize=3)
    # m_log_alphas = -np.log10(lassoCV_model.alphas_)
    # plt.errorbar(m_log_alphas, lassoCV_model.coef_, yerr=lassoCV_model.std_err_,
    #              fmt='o', ecolor='g', capsize=3)
    plt.axvline(lassoCV_model.alpha_, color='red', ls='--')  # 加一条垂直线
    plt.title('Errorbar')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.show()
    # # 此图显示随着lambda的变化，系数的变化走势
    # plt.plot(lassoCV_model.alphas, lassoCV_model.coef_, '-')
    # plt.axvline(lassoCV_model.alpha_, color='red', ls='--')
    # plt.xlabel('Lambda')
    # plt.ylabel('coef')
    # plt.show()