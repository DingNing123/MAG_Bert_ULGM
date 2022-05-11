"""
AIO -- All Model in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal
import pdb

from models.subNets.AlignNets import AlignSubNet
from models.multiTask import *

pretrained_weights = '/media/dn/newdisk/tools/bert-base-uncased/'

__all__ = ['AMIO']

MODEL_MAP = {
    'self_mm': SELF_MM, 
    'svm': SVM,
    'mult': MULT,
    # 'mag_bert': MAG_BERT,
}

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.need_model_aligned = args.need_model_aligned
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args.seq_lens = self.alignNet.get_seq_len()
        # 如果是mag_bert,加载预训练权重
        if args.modelName == 'mag_bert':
            args.beta_shift = 1.0
            args.dropout_prob = 0.5
            multimodal_config = MultimodalConfig(
                beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
            )
            self.Model = MAG_BertForSequenceClassification.from_pretrained(
                 pretrained_weights, multimodal_config=multimodal_config, args=args, num_labels=1,
            )
        else:
            lastModel = MODEL_MAP[args.modelName]
            self.Model = lastModel(args)


    def forward(self, text_x, audio_x, video_x):
        # import ipdb;ipdb.set_trace()
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x)

