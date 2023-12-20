#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
# import tensorflow as tf
import pandas as pd
import torch 
# pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
import seaborn as sns
from src.models.model_mega_hp import Model_MEGA_HP
from config import Config
from bertviz import head_view

def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png'):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    attention = attention[:15,:15].detach().numpy()
    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()


# Token化工具
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
config = Config()
path = "/home/zwb521/practice/useful_punc/all_debert/res/check_points/microsoft-deberta-v3-large/"
model = ParallelEndecoderGraph(config)
checkpoint_to_load = torch.load(path+"parallelendecoderbreezy-plum-impala.pt")
model.load_state_dict(checkpoint_to_load['model'])
model.eval()

sentence = 'How Attention works in Deep Learning understanding the attention mechanism in sequence models'
encoding = tokenizer(sentence, padding=True, truncation=True, max_length=256, return_tensors="pt")

inputs = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

inputs = torch.cat([inputs, torch.zeros((1, 256 - inputs.size(1)), dtype=torch.long)], dim=1)
attention_mask = torch.cat([attention_mask, torch.zeros((1, 256 - attention_mask.size(1)), dtype=torch.long)], dim=1)

attention = model(inputs,attention_mask)
attention =attention[-1]
# Token ids
ids = np.array(inputs[0])[:15]
texts = tokenizer.convert_ids_to_tokens(ids)
attention_plot(attention[0,-1,:,:], annot=True, x_texts=texts, y_texts=texts, figsize=(15, 15), figure_path='./figures',
            figure_name='bert_attention_weight_head_{}.png'.format(2))
tmp = attention[:,:,:15,:15] 
my_tuple = (tmp,)
vis = head_view(my_tuple, texts,html_action='return')
with open("./figures/allhead_view_all.html", 'w') as file:
    file.write(vis.data)

