#!/usr/bin/env python
# -*- coding:utf-8 -*-

SPECIAL_TOKENS_DICT = {}
SPECIAL_TOKENS_DICT['roberta-base'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['roberta-large'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['funnel-transformer/large'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT['funnel-transformer/xlarge'] = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}
SPECIAL_TOKENS_DICT["microsoft/deberta-v2-xlarge"] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '[UNK]', 'pad_token': '[PAD]'}
SPECIAL_TOKENS_DICT["microsoft/deberta-v3-large"] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '[UNK]', 'pad_token': '[PAD]'}
SPECIAL_TOKENS_DICT['bert-base-chinese'] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '[UNK]',  'pad_token': '[PAD]'}
SPECIAL_TOKENS_DICT["nghuyong/ernie-3.0-base-zh"] = {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '[UNK]',  'pad_token': '[PAD]'}
