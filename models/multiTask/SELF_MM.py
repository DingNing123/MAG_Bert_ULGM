# self supervised multimodal multi-task learning network
import os
import sys
import collections

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

__all__ = ['SELF_MM']

class SELF_MM(nn.Module):
    def __init__(self, args):
        super(SELF_MM, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        # change 32 -> 64 args.video_out = 64 2022年 04月 02日 星期六 16:47:28 CST
        # in conig/config_regression.py:266
        # args.text_out = 768
        # args.audio_out = 16 
        # args.video_out = 64 changed 32->64
        # args.post_video_dim = 64 changed 32->64
        self.video_model = VideoSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=0.0)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out , args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # second level fusion modulle
        # args.post_fusion_dim = 128
        self.post_fusion_layer_4 = nn.Linear(args.post_fusion_dim + args.text_out + args.video_out + args.audio_out , args.post_fusion_dim)
        self.post_fusion_layer_5 = nn.Linear(64,64)
        self.post_fusion_layer_5_1 = nn.Linear(64,64)
        # layer 6 is same for 3 , layer 3 is useless 
        self.post_fusion_layer_6 = nn.Linear(64, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

        '''
        # 0405 attention inspired by ../mmsa/models/singleTask/MFN.py
        attInShape = 64
        h_att1 = 64 
        att1_dropout = 0.7 
        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)
        '''

        '''
        0413 early_fusion lstm 
        
        '''
        # self.norm = nn.BatchNorm1d(126) # 126 is text len with context  : 
        self.norm = nn.BatchNorm1d(45-1) # 45 is text len : torch.Size([32, 45, 768])
        # self.lstm = nn.LSTM(33 + 768 + 709, 64, num_layers=1, dropout=0.0, bidirectional=False, batch_first=True)
        self.lstm = nn.LSTM(33 + 768 + 2048, 64, num_layers=1, dropout=0.0, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(64,64)
        self.out = nn.Linear(64,1)

        


    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()

        # 0412 align subnet 
        text_x = self.text_model(text)[:,1:,:]
        self.dst_len = text_x.size(1) # 45
        audio_x = audio
        video_x = video 

        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        text_x = align(text_x)
        audio_x = align(audio_x)
        video_x = align(video_x)

        # 0413 Early_fusion LSTM
        x = torch.cat([text_x, audio_x, video_x], dim=-1)
        # import ipdb;ipdb.set_trace()
        x = self.norm(x)
        _, final_states = self.lstm(x)
        x = self.dropout(final_states[0][-1].squeeze(dim=0))
        x = F.relu(self.linear(x), inplace=True)
        x = self.dropout(x)
        ef_lstm_h = x
        ef_lstm_output = self.out(x)

        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)

        '''
        0411 delete 64->64 self-attention
        # 0405 attention inspired by ../mmsa/models/singleTask/MFN.py
        attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(video)))),dim=1)
        cStar = video 
        attended = attention*cStar
        # residual 
        video = attended + video 
        '''
        
        # fusion 0415 放弃后期融合  
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)

        # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        # 0415 give up late fusion 
        '''
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        x_f = torch.cat([x_f, text, audio, video], dim=-1)
        # layer_3 is useless 
        # output_fusion = self.post_fusion_layer_3(x_f)
        x_f = F.relu(self.post_fusion_layer_4(x_f), inplace=False)
        '''
        # after layer_4 save x_f as fusion_h for unimodal label generation 0403
        # 0414 
        # 只是用text模态，考虑bert模型的性能。
        # x_f = text 
        # fusion_h = x_f 
        # x_f = torch.cat((ef_lstm_h,text),dim=-1)
        # 0415 修改为只使用ef-lstm的情况。
        x_f = ef_lstm_h
        # 0415 ulgm
        # fusion_h = x_f
        
        x_f = self.post_fusion_dropout(x_f)
        x_f = F.relu(self.post_fusion_layer_5(x_f), inplace=False)
        x_f = F.relu(self.post_fusion_layer_5_1(x_f), inplace=False)
        output_fusion = self.post_fusion_layer_6(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        # output_fusion出现nan情况,查看原因
        if output_fusion.isnan().any():
            pdb.set_trace()

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': x_f,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        # pdb.set_trace()
        # mustard数据集样本长短差别极大,mean+3*std依然有
        # 被截断的样本存在,因此需要更新送入rnn的长度.否则
        # 会运行错误
        '''
        在预处理阶段进行了bug的修复,更新了长度
        max_length = x.shape[1]
        batch_size = x.shape[0]
        condition = lengths <= max_length
        lengths = torch.where(condition,lengths,torch.ones(batch_size,).to(x.device).long()*max_length)
        '''

        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        try:
            _, final_states = self.rnn(packed_sequence)
        except RuntimeError:
            pdb.set_trace()
            print('Error')
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class VideoSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        0409 dingning add attention in to VideoSubNet 
        2048->128
        '''
        super(VideoSubNet, self).__init__()
        self.rnn1 = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.rnn2 = nn.LSTM(64, 64, num_layers=1, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # self.linear1 = nn.Linear(709, 64)
        self.linear1 = nn.Linear(2048,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        try:
            _, final_states = self.rnn1(packed_sequence)
        except RuntimeError:
            pdb.set_trace()
            print('Error')
        #  0409 add attention
        x1 = self.linear1(x)
        attention = F.softmax(self.linear2(self.dropout1(F.relu(x1))),dim=-1)
        # attended:(batch_size,sequence_len,64)
        attended = attention * x1
        h0 = final_states[0].squeeze()
        # h1:(batch_size,1,64)
        h1 = torch.unsqueeze(h0,1)
        # h2:(batch_size,sequence_len+1,64)
        h2 = torch.cat((h1,attended),1)
        # this line need debug add 1 
        lengths1 = lengths + 1
        packed_sequence1 = pack_padded_sequence(h2, lengths1, batch_first=True, enforce_sorted=False)
        _, final_states1 = self.rnn2(packed_sequence1)
        h3 = self.dropout2(final_states1[0].squeeze())
        
        h4 = self.dropout2(final_states[0].squeeze())
        h5 = h3 + h4
        # h:(batch_size,64)
        y_1 = self.linear3(h5)
        return y_1
