# self supervised multimodal multi-task learning network
# 融合两个Bert到一个Bert，共享参数，但是试验之后发现，性能并不如两个bert各有各的参数效果好。
# 6月17日
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
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler

import wandb

# Mustard SETTING
ACOUSTIC_DIM = 33
# 最优的实验是在2048维特征中得到的。
VISUAL_DIM = 2048
# 上下文的消融实验是在709维的特征下做的。
# VISUAL_DIM = 709
TEXT_DIM = 768

# MOSI SETTING
#ACOUSTIC_DIM = 5
#VISUAL_DIM = 20
#TEXT_DIM = 768

# __all__ = ['SELF_MM']
class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(MAG, self).__init__()
        print(
            "Initializing MAG with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )

        self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)

        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        # import ipdb;ipdb.set_trace()
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        DEVICE = visual.device

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)

        alpha = torch.min(thresh_hold, ones)

        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output


class MAG_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.MAG = MAG(
            config.hidden_size,
            multimodal_config.beta_shift,
            multimodal_config.dropout_prob,
        )

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        singleTask = False,
    ):
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Early fusion with MAG 
        # 如果是单任务，则不通过MAG门控网络
        # import ipdb;ipdb.set_trace()
        if singleTask:
            fused_embedding = embedding_output
        else:
            fused_embedding = self.MAG(embedding_output, visual, acoustic)


        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # import ipdb;ipdb.set_trace()

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        # 单任务提取pooled_output
        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

class MAG_BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, args=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MAG_BertModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        audio_in, video_in = args.feature_dims[1:]

        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = VideoSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        self.post_fusion_dropout = nn.Dropout(p=0.0)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out , args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)
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
        self.args = args

        self.init_weights()

    def forward(
        self,
        input_ids,
        acoustic,
        visual,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # 因为在模型对齐的代码中进行了转置，因此这里调整过来
        input_ids = input_ids.transpose(1,2)
        # import ipdb;ipdb.set_trace()
        input_ids_voc = input_ids[:,0,:].long() # torch.Size([32, 45])

        # 6月17日，融合多任务模型中的两个bert为1个，共享参数，根据经验，可以提升性能
        text_outputs = self.bert(
            input_ids_voc , 
            visual,
            acoustic,
            # visual = None,  
            # acoustic = None ,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            singleTask = True
        )
        text = text_outputs[1]
        
        # text = self.text_model(input_ids)[:,0,:]
        mask_len = torch.sum(input_ids[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        audio = self.audio_model(acoustic, text_lengths)
        video = self.video_model(visual, text_lengths)

        # import ipdb;ipdb.set_trace()

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

        # import ipdb;ipdb.set_trace()

        # 因为进行了模型的对齐，因此不再具备序列长度的元组记录，注释以下语句
        # visual = visual[0]
        # acoustic = acoustic[0]
        # import ipdb;ipdb.set_trace()

        outputs = self.bert(
            input_ids_voc,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            singleTask = False,
        )

        pooled_output = outputs[1]
        # pooled_output: torch.Size([48, 768])
        pooled_output = self.dropout(pooled_output)

        x = self.classifier(pooled_output)
        # speed is a 61-dimensional vector

        logits = x

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here


        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # import ipdb;ipdb.set_trace()
        output_fusion = outputs[0]# torch.Size([32, 1])

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': pooled_output,
        }
        return res

        # return outputs



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
        # self.linear1 = nn.Linear(709,64) # mustard openface2.0
        # self.linear1 = nn.Linear(20,64) # mosi
        self.linear1 = nn.Linear(2048,64) # mustard
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # import ipdb;ipdb.set_trace()
        _, final_states = self.rnn1(packed_sequence)
        #  0409 add attention
        # import ipdb;ipdb.set_trace()
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
