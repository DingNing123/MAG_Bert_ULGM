'''
生成mustard数据集的csv格式标签文件 Contextual 
讲话者独立 平衡的数据集 文献[1]平衡的划分,讲话者独立
'''
# coding: utf-8
import os
import shutil
import pickle
import librosa
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import pdb
import csv
import json

import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from globalstr import bert_en_path,bert_cn_path
from utils.functions import csv_header,csv_add_one_row,get_file_size

LABEL_CSV = 'labelContextIndep.csv'

class MDataPreLoader(Dataset):
    def __init__(self, args):
        self.working_dir = args.working_dir
        # pdb.set_trace()
        if args.mode == 'debug' :
            self.df = args.df[:20] # for debug
        else:
            self.df = args.df
        self.annotation_dict = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }
        # toolkits path
        self.openface2Path = args.openface2Path
        # bert
        tokenizer_class = BertTokenizer
        if args.language == 'cn':
            self.pretrainedBertPath = bert_cn_path 
            self.tokenizer = tokenizer_class.from_pretrained(bert_cn_path)
            self.model = BertModel.from_pretrained(self.pretrainedBertPath)
        else:
            self.pretrainedBertPath = bert_en_path 
            self.tokenizer = tokenizer_class.from_pretrained(bert_en_path, do_lower_case=True)
            self.model = BertModel.from_pretrained(self.pretrainedBertPath)
    
    def __len__(self):
        return len(self.df)

    def __getVideoEmbedding(self, video_path, tmp_dir, pool_size=3):
        # pdb.set_trace()
        faces_feature_dir = os.path.join(tmp_dir, 'Faces')
        # 如果已经生成了Faces目录 则不用再重复执行代码
        
        if not os.path.exists(faces_feature_dir) :
            os.mkdir(faces_feature_dir)
            cmd = self.openface2Path + ' -f ' + video_path + ' -out_dir ' + faces_feature_dir
            os.system(cmd)

        # read features
        features, local_features = [], []
        df_path = glob(os.path.join(faces_feature_dir, '*.csv'))
        if len(df_path) > 0:
            df_path = df_path[0]
            df = pd.read_csv(df_path)
            for i in range(len(df)):
                # 为了避免特征出现nan错误,因此不读取长度不够709维的帧特征,but 存在bug 读取到面部的视频帧过少 特征得到空,
                # to simplely handle this problem,use numpy.nan_to_num make nan to zero ,Do not to divide to 2 situations.
                # i.e. not think pool = 3 situation .
                # if torch.Tensor(df.loc[i][df.columns[5:]]).isnan().any()!=True:
                local_features.append(np.nan_to_num(np.array(df.loc[i][df.columns[5:]])))

                if (i + 1) % pool_size == 0:
                    features.append(np.array(local_features).mean(axis=0))
                    local_features = []

            if len(local_features) != 0:
                features.append(np.array(local_features).mean(axis=0))

        # 应该避免特征完全舍弃
        if len(np.array(features))==0:
            pdb.set_trace()
            # use numpy.nan_to_num to fix the bug

        return np.array(features)

    def __getAudioEmbedding(self, video_path, audio_path):
        # pdb.set_trace()
        # use ffmpeg to extract audio
        if not os.path.exists(audio_path) :
            cmd = 'ffmpeg -i ' + video_path + ' -f wav -vn ' + \
                    audio_path + ' -loglevel quiet'
            os.system(cmd)
        # get features
        y, sr = librosa.load(audio_path)
        # using librosa to get audio features (f0, mfcc, cqt)
        hop_length = 512 # hop_length smaller, seq_len larger
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T # (seq_len, 1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T # (seq_len, 20)
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T # (seq_len, 12)

        return np.concatenate([f0, mfcc, cqt], axis=-1)
    
    def __getTextEmbedding(self, text):
        # pdb.set_trace()
        # directory is fine
        # tokenizer = BertTokenizer.from_pretrained(self.pretrainedBertPath)
        tokenizer = self.tokenizer
        model = self.model
        # add_special_tokens will add start and end token
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze().numpy()
    
    def __preTextforBert(self, text):
        # pdb.set_trace()
        tokens_a = self.tokenizer.tokenize(text)
        # 我猜是因为transformer版本的原因,取消了invertable参数
        # tokens_a = self.tokenizer.tokenize(text,invertable=True)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        
        input_ids = np.expand_dims(input_ids, 1)
        input_mask = np.expand_dims(input_mask, 1)
        segment_ids = np.expand_dims(segment_ids, 1)

        text_bert = np.concatenate([input_ids, input_mask, segment_ids], axis=1)

        return text_bert

    def __getitem__(self, index):
        tmp_dir = os.path.join(self.working_dir, f'Self-MM-Processed/{self.df.loc[index]["video_id"]}')
        tmp_dir_c = tmp_dir + '_c'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if not os.path.exists(tmp_dir_c):
            os.makedirs(tmp_dir_c)
        video_id, clip_id, text, label, annotation, mode, _ = self.df.loc[index]
        cur_id = video_id + '$_$' + clip_id
        # video
        video_path = os.path.join(self.working_dir, 'utterances_final', video_id + '.mp4')
        #c context
        video_path_c = os.path.join(self.working_dir, 'context_final', video_id + '_c.mp4')
        
        embedding_V = self.__getVideoEmbedding(video_path, tmp_dir)
        embedding_V_c = self.__getVideoEmbedding(video_path_c, tmp_dir_c)
        embedding_V_a = np.concatenate((embedding_V_c,embedding_V))

        seq_V = embedding_V_a.shape[0]
        # audio
        audio_path = os.path.join(tmp_dir, 'tmp.wav')
        audio_path_c = os.path.join(tmp_dir_c, 'tmp.wav')

        embedding_A = self.__getAudioEmbedding(video_path, audio_path)
        embedding_A_c = self.__getAudioEmbedding(video_path_c, audio_path_c)
        embedding_A_a = np.concatenate((embedding_A_c,embedding_A))

        seq_A = embedding_A_a.shape[0]
        # text
        embedding_T = self.__getTextEmbedding(text) # (len,768)
        text_bert = self.__preTextforBert(text)# (len,3)
        seq_T = embedding_T.shape[0]# len

        ret = {
            'id': cur_id,
            'audio': embedding_A_a,
            'vision': embedding_V_a,
            'raw_text': text,
            'text': embedding_T,
            'text_bert': text_bert,
            'audio_lengths': seq_A,
            'vision_lengths': seq_V,
            'annotations': annotation,
            'classification_labels': self.annotation_dict[annotation],
            'regression_labels': label,
            'mode': mode
        }
        # clear tmp dir to save space
        # shutil.rmtree(tmp_dir)
        return ret

class MDataPre():
    def __init__(self, args):
        self.working_dir = args.working_dir
        # padding
        self.padding_mode = 'zeros'
        self.padding_location = 'back'
    
    def __padding(self, feature, MAX_LEN):
        """
        mode: 
            zero: padding with 0
            normal: padding with normal distribution
        location: front / back
        """
        assert self.padding_mode in ['zeros', 'normal']
        assert self.padding_location in ['front', 'back']

        length = feature.shape[0]
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :]
        
        if self.padding_mode == "zeros":
            pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
        elif self.padding_mode == "normal":
            mean, std = feature.mean(), feature.std()
            pad = np.random.normal(mean, std, (MAX_LEN-length, feature.shape[1]))

        feature = np.concatenate([pad, feature], axis=0) if(self.padding_location == "front") else \
                  np.concatenate((feature, pad), axis=0)
        return feature

    def __paddingSequence(self, sequences):
        # pdb.set_trace()
        if len(sequences) == 0:
            return sequences
        feature_dim = sequences[0].shape[-1]
        lens = [s.shape[0] for s in sequences]
        # confirm length using (mean + std)
        final_length = int(np.mean(lens) + 3 * np.std(lens))
        # padding sequences to final_length
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        for i, s in enumerate(sequences):
            final_sequence[i] = self.__padding(s, final_length)

        return final_sequence

    def __collate_fn(self, batch):
        ret = {k: [] for k in batch[0].keys()}
        for b in batch:
            for k,v in b.items():
                ret[k].append(v)
        return ret
    
    def run(self):
        if args.mode == 'debug':
            output_path = os.path.join(self.working_dir, 'Self-MM-Processed/featuresContext20.pkl')
        else:
            output_path = os.path.join(self.working_dir, 'Self-MM-Processed/featuresContextIndep.pkl')
        # load last point
        # 不再考虑从中间继续执行的情况,每次均重新生成,不再load last point
        # 也即,不再考虑output_path是否存在
        '''
        if os.path.exists(output_path):
            # pdb.set_trace()
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
            last_row_idx = np.sum([len(data[mode]['id']) for mode in ['train', 'valid', 'test']])
            # 调试目的 暂时将last_row_ids 设为0
            last_row_idx = 0
        else:
        '''
        # 每次重新生成数据逻辑更简单,关注更能出成果的模型修改部分 
        data = {"id": [], 
                "raw_text": [],
                "audio": [],
                "vision": [],
                "text": [],
                "text_bert": [],
                "audio_lengths": [],
                "vision_lengths": [],
                "annotations": [],
                "classification_labels": [], 
                "regression_labels": [],
                "mode": []}
        last_row_idx = 0

        if args.mode == 'debug':
            args.df = pd.read_csv(os.path.join(self.working_dir, 'labelContext20.csv'), dtype={'clip_id': str, 'video_id': str, 'text': str})
        else:
            args.df = pd.read_csv(os.path.join(self.working_dir, LABEL_CSV ), dtype={'clip_id': str, 'video_id': str, 'text': str})

        args.df = args.df[last_row_idx:]

        dataloader = DataLoader(MDataPreLoader(args),
                                batch_size=4,
                                num_workers=0,
                                shuffle=False,
                                collate_fn=self.__collate_fn)
        isEnd = False
        #try:
        with tqdm(dataloader) as td:
            for batch_data in td:
                for k, v in batch_data.items():
                    data[k].extend(v)
        # 假设肯定执行成功,
        isEnd = True
        #except Exception as e:
            #print(e)
        #finally:
        #try:
        if isEnd:
            # padding
            for item in ['audio', 'vision', 'text', 'text_bert']:
                data[item] = self.__paddingSequence(data[item])
                if item == 'text_bert':
                    data[item] = data[item].transpose(0,2,1)

            # split train, valid, test
            
            inx_dict = {
                mode + '_index': [i for i, v in enumerate(data['mode']) if v == mode]
                for mode in ['train', 'valid', 'test']
            }
            data.pop('mode')
            final_data = {k: {} for k in ['train', 'valid', 'test']}
            for mode in ['train', 'valid', 'test']:
                indexes = inx_dict[mode + '_index']
                # pdb.set_trace()
                for item in data.keys():
                    if isinstance(data[item], list):
                        final_data[mode][item] = [data[item][v] for v in indexes]
                    else:
                        final_data[mode][item] = data[item][indexes]
            data = final_data
        #except Exception as e:
        #    print(e)
        #finally:
            # pdb.set_trace()
            # 为了校正错误: 任何一个样本的长度不应该超过截断的最大长度
        for mode in ['train','valid','test']:
            for single in ['audio','vision']:
                max_length = data[mode][single].shape[1]
                single_lengths = single + '_lengths'
                data[mode][single_lengths]=torch.where(torch.tensor(data[mode][single_lengths]) <= max_length,
                        torch.tensor(data[mode][single_lengths]),max_length).tolist()
                assert max(data[mode][single_lengths]) <= max_length

        with open(output_path, 'wb') as wf:
            pickle.dump(data, wf, protocol = 4)

        print('Features are saved in %s!' %output_path)
        # 读取生成文件大小,以人类友好的方式显示
        get_file_size(output_path,"MB")

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default='/media/dn/newdisk/datasets/mmsd_raw_data/',
                        help='path to datasets')
    parser.add_argument('--mode', type=str, default="run",
            help='run:use all entries / debug : means use 20 entries')
    parser.add_argument('--language', type=str, default="en",
                        help='en / cn')
    parser.add_argument('--openface2Path', type=str, default="/media/dn/home/dn/1.tools/OpenFace/build/bin/FeatureExtraction",
                        help='path to FeatureExtraction tool in openface2')
    return parser.parse_args()

def clean_text_remove_punctuation(text):
    '''
    清理文本中的标点符号,并且转换为大写
    '''
    punctuation = '!,;:?"、，；.'
    import re
    text1 = re.sub(r'[{}]+'.format(punctuation),' ',text)
    text2 = re.sub(r'[\']','',text1)
    text2 = re.sub(r'[\']','',text2)
    text3 = re.sub(r'[\n]',' ',text2)
    return text3.strip().upper()

def from_json_to_entry(ID,dataset_json):
    video_id = ID
    clip_id = 0
    text = dataset_json[ID]['utterance']
    textContext = dataset_json[ID]['context']
    textContext = ','.join(textContext)

    text = clean_text_remove_punctuation(text)
    textContext = clean_text_remove_punctuation(textContext)
    text = textContext + ' ' + text

    label = 1.0 if dataset_json[ID]['sarcasm'] else -1.0
    annotation = 'Positive' if dataset_json[ID]['sarcasm'] else 'Negative'
    label_by = 0
    return video_id, clip_id, text, label, annotation, label_by  

def gen_label_csv(args):
    '''
    生成mustard数据集的csv格式标签文件 Contextual speaker independent train_5.csv
    /media/dn/newdisk/datasets/mmsd_raw_data/labelContext.csv
    '''
    label_csv = os.path.join(args.working_dir,LABEL_CSV  )
    fieldnames = ['video_id','clip_id','text', 'label', 'annotation',
            'mode', 'label_by']

    csv_header(label_csv,fieldnames)

    DATA_PATH_JSON = os.path.join(args.working_dir, 'sarcasm_data.json')
    dataset_json = json.load(open(DATA_PATH_JSON))

    # train_index0.csv is  5折交叉 speaker dependent 
    train_index_file = os.path.join(args.working_dir, 'csv/train_index5.csv')
    test_index_file = os.path.join(args.working_dir, 'csv/test_index5.csv')
    train_index = np.array(pd.read_csv(train_index_file)).reshape(-1)
    test_index = np.array(pd.read_csv(test_index_file)).reshape(-1)

    for idx, ID in enumerate(list(dataset_json.keys())[:]):
        # clean the code to func video_id,clip_id,...,annotation
        video_id, clip_id, text, label, annotation, label_by = from_json_to_entry(ID,dataset_json)
        # train valid test
        mode = 'train' if idx in train_index else 'valid'
        row = {'video_id':video_id,
                'clip_id':clip_id,
                'text':text,
                'label':label,
                'annotation':annotation,
                'mode':mode,
                'label_by':label_by,
                }

        csv_add_one_row(label_csv, fieldnames, row)

    for idx, ID in enumerate(list(dataset_json.keys())[:]):
        # train valid test
        mode = 'train' if idx in train_index else 'test'
        if mode == 'test' : 
            video_id, clip_id, text, label, annotation, label_by = from_json_to_entry(ID,dataset_json)
            row = {'video_id':video_id,
                    'clip_id':clip_id,
                    'text':text,
                    'label':label,
                    'annotation':annotation,
                    'mode':mode,
                    'label_by':label_by }
            csv_add_one_row(label_csv, fieldnames, row)
    print(f'saved in {label_csv}')
    exit()


if __name__ == "__main__":
    args = parse_args()
    # 去掉下面的注释,生成标签文件划分训练验证集集
    # 5倍交叉的话 需要重复生成 需要时间  
    # gen_label_csv(args)

    dp = MDataPre(args)
    dp.run()
