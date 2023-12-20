#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from transformers import AutoModel,AutoTokenizer
from transformers import MegaConfig, MegaModel

class Model_MEGA_HP(nn.Module):
    """docstring for LanModelGraph"""
    def __init__(self, config):
        super(Model_MEGA_HP, self).__init__()
        conf = MegaConfig()
        conf.vocab_size = 128000 
        conf.num_hidden_layers = 8 
        conf.max_positions = 1024
        conf.hidden_size = 768
        conf.eos_token_id = 102  
        conf.bos_token_id = 101  
        conf.pad_token_id = 0     
        conf.output_attentions = True
        self.config = config
        self.decode_layer = MegaModel(conf)
        self.encode_layer = AutoModel.from_pretrained(config.lan_model, hidden_dropout_prob=config.xfmr_hidden_dropout_prob,output_attentions=True)

        fusion_in_features = config.lan_hidden_size + conf.hidden_size
        self.fusion_layer = nn.TransformerEncoderLayer(
             d_model=fusion_in_features
           , nhead=config.xfmr_num_attention_heads
           , dim_feedforward=config.xfmr_intermediate_size
           , dropout=config.xfmr_hidden_dropout_prob
           , activation='gelu'
              )

        self.dropout_layer = nn.Dropout(config.xfmr_hidden_dropout_prob)
        # freeze the Pre-trained Language Model
        if config.freeze_lan_model:
            for param in self.lan_layer.base_model.parameters():
                param.requires_grad = False

        # hp module
        self.mlp = torch.nn.Linear(fusion_in_features,3)
        self.text_to_label = torch.tensor([
                                    [-1,0,0],
                                    [1,1,0],
                                    [1,-1,1],
                                    [1,-1,-1]
                                    ]
                                    )
        self.bias = torch.tensor([
                                [1,1,1],
                                [0,0,1],
                                [0,1,0],
                                [0,1,1]
                                ]
                                )
        

    def forward(
            self,
            xs,
            x_masks,
            y_tags=None,
            y_mask=None,
            device = None
    ):
        xs_encode = self.encode_layer(xs, attention_mask=x_masks)[0]
        xs_decode = self.decode_layer(xs, attention_mask=x_masks)[0]
        # fusion 
        xs_cat = torch.cat((xs_encode, xs_decode), dim=-1)
        xs = self.fusion_layer(xs_cat)
        x = self.dropout_layer(xs)

    ###################
    # HP module
    ###################
        node_info = self.mlp(x) 
        node_info_4 = node_info.unsqueeze(2).repeat([1,1,4,1])
        # sigmod func
        node_info_4_sig = torch.sigmoid(node_info_4)
        # avoid the case sigmoid(h) == 0 or 1 when |h| > 10
        eps = 1e-9
        node_info_4_sig = torch.clip(node_info_4_sig, eps, 1 - eps)
        # sign bias
        node_output = node_info_4_sig*self.text_to_label.to(device) + self.bias.to(device)
        node_output_log = torch.log(node_output)
        log_prob_4 = torch.sum(node_output_log,-1)
        return log_prob_4




