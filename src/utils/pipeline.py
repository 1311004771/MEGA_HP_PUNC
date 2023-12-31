#!/usr/bin/env python
# -*- coding:utf-8 -*-

# dependency
import os
import random
# public
from tqdm import tqdm, trange
import torch
from torch.utils import data as torch_data
# private
from ..utils import save, load
from ..models import  model_mega_hp


def read_file(path: str, config) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    sents_list = []
    sent = []
    for i in range(len(lines)):
    # for i in range(100):
        l = lines[i].split()
        if len(l) == 1:
            l = [config.UNK_TOKEN] + l
        sent.append('\t'.join(l))
        if l[-1] in {config.PERIOD_TOKEN, config.QUESTION_TOKEN}:
            sents_list.append(sent)
            sent = []
    return sents_list

def parse_data(sents_list: list, tokenizer, config):
    for i in trange(len(sents_list)):
        sent = sents_list[i]
        x, y, y_mask = [], [], []
        for token_pun_pair in sent:
            word, punc = token_pun_pair.split('\t')
            tokens = tokenizer.tokenize(word)
            for j in range(len(tokens) - 1):
                x.append(tokens[j])
                y.append(config.NORMAL_TOKEN)
                y_mask.append(0)
            x.append(tokens[-1])
            y.append(punc)
            y_mask.append(1)
        sents_list[i] = [d for d in zip(x, y, y_mask)]
    # return sents_list
    return [s for sent in sents_list for s in sent]


def generate_seq(lines: list, config):
    x, y, y_mask = [config.BOS_TOKEN], [config.NORMAL_TOKEN], [0]
    for l in lines:
        # token, pun, mask, tag
        if len(l) == 4:
            token, pun, mask, _ = l
        else:
            token, pun, mask = l
        x.append(token)
        y.append(pun)
        y_mask.append(mask)
    x.append(config.EOS_TOKEN)
    y.append(config.NORMAL_TOKEN)
    y_mask.append(0)

    return (x, y, y_mask)


def generate_seqs(lines: list, config):
    xs, ys, y_masks = [], [], []

    x, y, y_mask = [config.BOS_TOKEN], [config.NORMAL_TOKEN], [0]
    for l in lines:
        # token, pun, mask, tag
        if len(l) == 4:
            token, pun, mask, _ = l
        else:
            token, pun, mask = l
        if len(x) >= config.max_seq_len - 1:
            x.append(config.EOS_TOKEN)
            y.append(config.NORMAL_TOKEN)
            y_mask.append(0)
            xs.append(x)
            ys.append(y)
            y_masks.append(y_mask)
            x, y, y_mask = [config.BOS_TOKEN], [config.NORMAL_TOKEN], [0]

        else:
            x.append(token)
            y.append(pun)
            y_mask.append(mask)
    if len(x) > 1:
        x.append(config.EOS_TOKEN)
        y.append(config.NORMAL_TOKEN)
        y_mask.append(0)
        xs.append(x)
        ys.append(y)
        y_masks.append(y_mask)

    return (xs, ys, y_masks)

class Dataset(torch_data.Dataset):
    """docstring for Dataset"""
    def __init__(self, file, config, is_train=False, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.is_train = is_train
        data_file = config.DATA_JSON.format(file)
        self.raw_data = load.load_json(data_file)
        if not training:
            self.raw_data = self.raw_data[:51200]
        if self.is_train and self.config.seq_sampling:
            self.data_size = len(self.raw_data) // self.config.max_seq_len
            self.sample_pool = set(range(len(self.raw_data)-config.max_seq_len-1))
        else:
            self.get_pairs()
            self.data_size = len(self.xs)

    def get_pairs(self):
        self.xs, self.ys, self.y_masks = generate_seqs(self.raw_data, self.config)

    def __len__(self): 
        return self.data_size

    def __getitem__(self, idx): 
        if self.is_train and self.config.seq_sampling:
            if not self.sample_pool:
                self.sample_pool = set(range(len(self.raw_data)-self.config.max_seq_len-1))
            idx = random.choice(list(self.sample_pool))
            self.sample_pool.remove(idx)
            return generate_seq(self.raw_data[idx:idx+self.config.max_seq_len-2], self.config)
        else:
            return self.xs[idx], self.ys[idx], self.y_masks[idx]

def translate(seq: list, trans_dict: dict) -> list: 
    return [trans_dict[token] for token in seq]

def count_parameters(model): 
    # get total size of trainable parameters 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pick_model(config):
    if config.lan_model:
        if config.model_name == 'MEGA_HP':
            return model_mega_hp.Model_MEGA_HP(config).to(config.device)
        else:
            raise ValueError('Wrong model to pick.')
    else:
        raise ValueError('Wrong language model to pick.')

def post_process(xs, x_masks, ys, y_masks, ys_, tokenizer, config):

    xs, x_masks, ys = (i.cpu().detach().numpy().tolist() for i in (xs, x_masks, ys))
    ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy().tolist()
    xs_lens = [sum(x_mask) for x_mask in x_masks]
    xs = [x[:l] for x, l in zip(xs, xs_lens)]
    ys = [y[:l] for y, l in zip(ys, xs_lens)]
    y_masks = [y_mask[:l] for y_mask, l in zip(y_masks, xs_lens)]
    ys_ = [y_[:l] for y_, l in zip(ys_, xs_lens)]
    xs = [tokenizer.convert_ids_to_tokens(x) for x in xs]
    ys = [translate(y, config.idx2label_dict) for y in ys]
    ys_ = [translate(y_, config.idx2label_dict) for y_ in ys_]

    return xs, ys, y_masks, ys_

def show_config(config, model):
    # general information
    general_info = '\n*Configuration*'
    general_info += '\nmodel: {}'.format(config.model_name)
    general_info += '\nlanguage model: {}'.format(config.lan_model)
    general_info += '\nfreeze language model: {}'.format(config.freeze_lan_model)
    general_info += '\nsequence boundary sampling: {}'.format(config.seq_sampling)
    general_info += '\nmask loss: {}'.format(config.mask_loss)
    general_info += '\ntrainable parameters: {:,.0f}'.format(config.num_parameters)
    model_info = '\nmodel:'
    for parameters in model.state_dict():
        model_info += '\n{}\t{}'.format(parameters, model.state_dict()[parameters].size())
    general_info += model_info
    general_info += '\ndevice: {}'.format(config.device)
    general_info += '\ntrain size: {}'.format(config.train_size)
    general_info += '\nval size: {}'.format(config.valid_size)
    general_info += '\nref test size: {}'.format(config.ref_test_size)
    general_info += '\nasr test size: {}'.format(config.asr_test_size)
    general_info += '\nbatch size: {}'.format(config.batch_size)
    general_info += '\ntrain batch: {}'.format(config.train_batch)
    general_info += '\nval batch: {}'.format(config.valid_batch)
    general_info += '\nref test batch: {}'.format(config.ref_test_batch)
    general_info += '\nasr test batch: {}'.format(config.asr_test_batch)
    general_info += '\nvalid win size: {}'.format(config.valid_win_size)
    general_info += '\nif load check point: {}'.format(config.load_check_point)
    if config.load_check_point:
        general_info += '\nModel restored from {}'.format(config.SAVE_POINT)
    general_info += '\n'
    print(general_info)

    return general_info

def save_model(step, epoch, model_state_dict, path):
    # save model, optimizer, and everything required to keep
    checkpoint_to_save = {
        'step': step,
        'epoch': epoch,
        'model': model_state_dict()}
    torch.save(checkpoint_to_save, path)
    print('Model saved as {}.'.format(path))

def save_model_opt(step, epoch, model_state_dict, opt_state_dict, path):
    # save model, optimizer, and everything required to keep
    checkpoint_to_save = {
        'step': step,
        'epoch': epoch,
        'model': model_state_dict(),
        'optimizer': opt_state_dict()}
    torch.save(checkpoint_to_save, path)
    print('Model saved as {}.'.format(path))
