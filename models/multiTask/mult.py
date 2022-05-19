# mult.py
# ding ning 754406193@qq.com
# 2022年 04月 24日 星期日 18:38:30 CST
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder
from models.subNets.transformers_encoder.transformer import TransformerEncoder

class MULT(nn.Module):
    def __init__(self,args):
        super(MULT,self).__init__()
        self.aligned = args.need_data_aligned

        self.proj_l = nn.Conv1d(768,30,kernel_size=1,padding=0,bias=False)
        self.proj_a = nn.Conv1d(33,30,kernel_size=3,padding=0,bias=False)
        self.proj_v = nn.Conv1d(2048,30,kernel_size=3,padding=0,bias=False)

        self.trans_l_with_a = TransformerEncoder(embed_dim=30,num_heads=10,layers=4,attn_dropout=0.0,relu_dropout=0.1,res_dropout=0.1,embed_dropout=0.25,attn_mask=True)
        self.trans_l_with_v = TransformerEncoder(embed_dim=30,num_heads=10,layers=4,attn_dropout=0.0,relu_dropout=0.1,res_dropout=0.1,embed_dropout=0.25,attn_mask=True)
        self.trans_l_mem    = TransformerEncoder(embed_dim=60,num_heads=10,layers=3,attn_dropout=0.0,relu_dropout=0.1,res_dropout=0.1,embed_dropout=0.25,attn_mask=True)

        self.proj1 = nn.Linear(60, 60)
        self.proj2 = nn.Linear(60, 60)
        self.out_layer = nn.Linear(60, 1)

        # 为了能匹配代码中的参数优化，设置声音子模型和图像子模型
        self.audio_model = AuViSubNet(33,16,16,num_layers=1,dropout=0.0)
        self.video_model = VideoSubNet(2048, 64,64,num_layers=1,dropout=0.0)
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=0.1)
        self.post_text_layer_1 = nn.Linear(768,64)
        self.post_text_layer_2 = nn.Linear(64,64)
        self.post_text_layer_3 = nn.Linear(64, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=0.1)
        self.post_audio_layer_1 = nn.Linear(16,16)
        self.post_audio_layer_2 = nn.Linear(16,16)
        self.post_audio_layer_3 = nn.Linear(16, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=0.1)
        self.post_video_layer_1 = nn.Linear(64,64)
        self.post_video_layer_2 = nn.Linear(64,64)
        self.post_video_layer_3 = nn.Linear(64, 1)

        self.post_fusion_dropout = nn.Dropout(p=0.0)
        self.post_fusion_layer_1 = nn.Linear(768 + 16 + 64, 64)

    def forward(self,text,audio,video):
        audio, audio_lengths = audio
        video, video_lengths = video
        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        # 0412 align subnet 
        text_x = self.text_model(text)[:,1:,:]
        self.dst_len = text_x.size(1) # 45

        x_l = text_x
        x_a = audio
        x_v = video 

        audio_x = audio
        video_x = video

        # training is only in train mode 
        x_l = F.dropout(x_l.transpose(1, 2), p=0.2, training=self.training)
        # Size([32, 45, 768])
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        # x_a.shape Size([32, 33, 645])
        # x_v.shape Size([32, 2048, 46])

        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) 
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)

        h_ls = self.trans_l_mem(h_ls)
        last_h_l = last_hs =  h_ls[-1]
        # h_ls.shape torch.Size([45, 32, 60])
        # h_ls[-1].shape torch.Size([32, 60])
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=0.1, training=self.training))
        last_hs_proj = last_hs_proj + last_hs

        output = self.out_layer(last_hs_proj)
        # ipdb> output.shape torch.Size([32, 1])

        # import ipdb;ipdb.set_trace()

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

        text = self.text_model(text)[:,0,:]

        # 对齐模型可以提高性能，因此不再进行参数控制 
        # 因为暂时不考虑bert文本微调，因此不对齐了。 2022年 04月 27日 星期三 18:29:44 CST
        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)

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
        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output,
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': last_hs_proj,
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
        self.linear1 = nn.Linear(2048,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn1(packed_sequence)
        # import ipdb;ipdb.set_trace()
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
