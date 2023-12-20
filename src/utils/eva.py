#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics

class Evaluater(object):
    """docstring for Evaluater"""
    def __init__(self, ys, y_masks, ys_, config):
        super(Evaluater, self).__init__()
        self.ys = [l for y in ys for l in y]
        self.ys_ = [l for y_ in ys_ for l in y_]
        self.y_masks = [m for y_mask in y_masks for m in y_mask]
        self.ys = self.compress(self.ys, self.y_masks)
        self.ys_ = self.compress(self.ys_, self.y_masks)
        self.labels = list(config.label2idx_dict.keys())[1:]

        self.comma_precision, self.period_precision, self.question_precision = self.get_precision()
        self.comma_recall, self.period_recall, self.question_recall = self.get_recall()
        self.comma_f1, self.period_f1, self.question_f1 = self.get_f1()

        self.avg_precision = precision_score(self.ys, self.ys_, average='micro', zero_division=0., labels=self.labels)
        self.avg_recall = recall_score(self.ys, self.ys_, average='micro', zero_division=0., labels=self.labels)
        self.avg_f1 = 2 * (self.avg_precision * self.avg_recall) / (self.avg_precision + self.avg_recall)
        self.key_metric = self.avg_f1
        # generate an evaluation message
        comma_msg = 'COMMA P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.comma_precision, self.comma_recall, self.comma_f1)
        period_msg = 'PERIOD P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.period_precision, self.period_recall, self.period_f1)
        question_msg = 'QUESTION P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.question_precision, self.question_recall, self.question_f1)
        avg_msg = 'Overall P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.avg_precision, self.avg_recall, self.avg_f1)
        # key metric for early stopping
        self.eva_msg_micro = '\n' + comma_msg + '\n' + period_msg + '\n' + question_msg + '\n' + avg_msg  + '\n'

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            self.ys, self.ys_, average=None, labels=self.labels)
        overall = metrics.precision_recall_fscore_support(
            self.ys, self.ys_, average='macro', labels=self.labels)

        # generate an evaluation message
        comma_msg = 'COMMA P:{:.4f} R:{:.4f} F:{:.4f}'.format(precision[0],recall[0], f1[0])
        period_msg = 'PERIOD P:{:.4f} R:{:.4f} F:{:.4f}'.format(precision[1],recall[1], f1[1])
        question_msg = 'QUESTION P:{:.4f} R:{:.4f} F:{:.4f}'.format(precision[2],recall[2], f1[2])
        avg_msg = 'Overall P:{:.4f} R:{:.4f} F:{:.4f}'.format(overall[0], overall[1], overall[2])
        # key metric for early stopping
        self.eva_msg_macro = '\n' + comma_msg + '\n' + period_msg + '\n' + question_msg + '\n' + avg_msg
        self.eva_msg = self.eva_msg_micro + self.eva_msg_macro


    def compress(self, data, mask):
        return [d for d, s in zip(data, mask) if s]

    def get_precision(self):
        return precision_score(self.ys, self.ys_, average=None, zero_division=0., labels=self.labels)

    def get_recall(self):
        return recall_score(self.ys, self.ys_, average=None, zero_division=0., labels=self.labels)

    def get_f1(self):
        return f1_score(self.ys, self.ys_, average=None, zero_division=0., labels=self.labels)