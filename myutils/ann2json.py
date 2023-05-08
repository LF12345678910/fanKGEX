# encoding: utf-8
# @author: fan
# @file: ann2json.py
# @time: 2023/3/6 9:59

import pandas as pd

df = pd.read_csv('test.ann', sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)