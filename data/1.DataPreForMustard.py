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
from overrides import overrides

import torch
import torch.nn
import torchvision
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from globalstr import bert_en_path,bert_cn_path
from utils.functions import csv_header,csv_add_one_row,get_file_size
import PIL.Image


class MDataPreLoader(Dataset):
    def __init__(self, args):
        self.working_dir = args.working_dir
        # 0317 丁宁修改 elif runIndep
        if args.mode == 'debug' :
            self.df = args.df[:20] # for debug
        elif args.mode == 'run':
            self.df = args.df
        elif args.mode == 'runIndep' or args.mode == 'runIndepResnet152':
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
        # ding ning 0325 for resnet 152 feature add args 
        self.args = args
        # 加载模型 resnet152 0326
        if args.mode == 'runIndepResnet152':
            self.resnet = self.pretrained_resnet152()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.transform = transforms
    
    def __len__(self):
        return len(self.df)

    def pretrained_resnet152(self) -> torch.nn.Module:
        resnet152 = torchvision.models.resnet152(pretrained=True)

        resnet152.eval() # Sets the module in evaluation mode
        for param in resnet152.parameters():
            param.requires_grad = False
        return resnet152

    def __getResnet152VideoEmbedding(self, frames_path):
        frames = None
        # 记录了当前视频有多少帧，即分割为几张图片
        frames_n = len(os.listdir(frames_path))
        for i, frame_file_name in enumerate(os.listdir(frames_path)):
            frame = PIL.Image.open(os.path.join(frames_path, frame_file_name))
            if self.transform:
                frame = self.transform(frame)
                # frame.shape torch.Size([3, 224, 224])
            if frames is None:
                frames = torch.empty((frames_n , *frame.size()))
            frames[i] = frame
        # (Pdb) p frames.shape torch.Size([14, 3, 224, 224]) torch - np.array 
        # 14,3,224,224 -> 14,2048 need resnet152 model 

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet = self.resnet.to(DEVICE)

        class Identity(torch.nn.Module):
            @overrides
            def forward(self, input_):
                return input_

        resnet.fc = Identity()  # Trick to avoid computing the fc1000 layer, as we don't need it here.
        # 0326 to device frames 14 3 224 224 
        frames = frames.to(DEVICE)
        batch_size = 32
        feature = torch.empty((frames_n,2048))
        for start_index in range(0, len(frames), batch_size):
            end_index = min(start_index + batch_size, len(frames))
            frame_ids_range = range(start_index, end_index)
            frame_batch = frames[frame_ids_range]
            avg_pool_value = resnet(frame_batch)
            # avg_pool_value is in cuda, remember .cpu()
            feature[frame_ids_range] = avg_pool_value.cpu()

        return np.array(feature)

    def __getVideoEmbedding(self, video_path, tmp_dir, pool_size=3):
        # 0322  Resnet152 丁宁 
        # 我意识到了只需要集成  
        # 不需要融合到代码中  
        # 读取已有特征
        # 处理为Self-MM 需要的模式即可  感觉还是应该集成代码
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
        # ding ning 0325 modified resnet 152 feature 
        tmp_dir = os.path.join(self.working_dir, f'Self-MM-Processed/{self.df.loc[index]["video_id"]}')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        video_id, clip_id, text, label, annotation, mode, _ = self.df.loc[index]
        cur_id = video_id + '$_$' + clip_id
        # video
        video_path = os.path.join(self.working_dir, 'utterances_final', video_id + '.mp4')
        
        # dingning 0325 resnet 152 modified 
        # 参考Mustard-master代码生成2048特征 resnet152
        # 设计输入输出 
        if self.args.mode == 'runIndepResnet152':
            frames_path = '/media/dn/newdisk/datasets/mmsd_raw_data/Processed/video/Frames/' 
            frames_path = frames_path + video_id
            embedding_V = self.__getResnet152VideoEmbedding(frames_path)
            seq_V = embedding_V.shape[0]
        else :
            embedding_V = self.__getVideoEmbedding(video_path, tmp_dir)
            seq_V = embedding_V.shape[0]

        # audio
        audio_path = os.path.join(tmp_dir, 'tmp.wav')
        embedding_A = self.__getAudioEmbedding(video_path, audio_path)
        seq_A = embedding_A.shape[0]
        # text
        embedding_T = self.__getTextEmbedding(text)
        text_bert = self.__preTextforBert(text)
        seq_T = embedding_T.shape[0]

        ret = {
            'id': cur_id,
            'audio': embedding_A,
            'vision': embedding_V,
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
        output_path = os.path.join(self.working_dir, 'Self-MM-Processed/features.pkl')
        # 0317 丁宁 修改 生成 独立特征 讲话者独立设置 
        if args.mode == 'runIndep':
            output_path = os.path.join(self.working_dir, 'Self-MM-Processed/featuresIndep.pkl')
        # 0322 丁宁 修改 生成 独立特征 讲话者独立设置  resnet152
        if args.mode == 'runIndepResnet152':
            output_path = os.path.join(self.working_dir, 'Self-MM-Processed/featuresIndepResnet152.pkl')

        # load last point
        # 不再考虑从中间继续执行的情况,每次均重新生成,不再load last point
        # 也即,不再考虑output_path是否存在
        # 0325 dingning delete 10 line code, make it clean and neat .  
        # 每次重新生成数据逻辑更简单,关注更能出成果的模型

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

        # 0317 丁宁 修改 讲话者独立设置 elif 
        if args.mode == 'debug':
            args.df = pd.read_csv(os.path.join(self.working_dir, 'label20.csv'), dtype={'clip_id': str, 'video_id': str, 'text': str})
        elif args.mode == 'run':
            args.df = pd.read_csv(os.path.join(self.working_dir, 'label.csv'), dtype={'clip_id': str, 'video_id': str, 'text': str})
        # 0322 丁宁 resnet152
        elif args.mode == 'runIndep' or args.mode == 'runIndepResnet152':
            args.df = pd.read_csv(os.path.join(self.working_dir, 'label_indep.csv'), dtype={'clip_id': str, 'video_id': str, 'text': str})

        args.df = args.df[last_row_idx:]

        dataloader = DataLoader(MDataPreLoader(args),
                                batch_size=64,
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
    # 丁宁修改20220322 规范源代码编写 
    # genLabelIndep : generate_label_csv
    # --mode genLabelIndep
    # --mode run 运行dependent设置
    # --mode debug 20条数据
    # --mode genLabelIndep 生成Indep标签
    # --mode genLabelDep 生成Dep标签
    # --mode runIndep 生成独立 openface 709 dim 特征 
    # --mode runIndepResnet152 独立 2048 dim resnet152 

    parser.add_argument('--mode', type=str, default="run", help='see source code')
    parser.add_argument('--language', type=str, default="en", help='en / cn')
    # 0322 openface to FeatureExtraction 可执行路径 
    parser.add_argument('--openface2Path', type=str, default="/media/dn/home/dn/下载/tools/OpenFace/build/bin/FeatureExtraction", help='')

    return parser.parse_args()

def clean_text_remove_punctuation(text):
    '''
    清理文本中的标点符号,并且转换为大写
    '''
    punctuation = '!,;:?"、，；.'
    import re
    text1 = re.sub(r'[{}]+'.format(punctuation),' ',text)
    text2 = re.sub(r'[\']','',text1)
    return text2.strip().upper()


def gen_label_csv(args):
    '''
    生成mustard数据集的csv格式标签文件
    /media/dn/newdisk/datasets/mmsd_raw_data/label.csv
    '''
    if args.mode == 'genLabelDep':
        label_csv = os.path.join(args.working_dir, 'label.csv')
    # 丁宁修改 20220315
    # 生成label_indep.csv label independent csv our split 平衡的划分 
    elif args.mode == 'genLabelIndep':
        label_csv = os.path.join(args.working_dir, 'label_indep.csv')

    fieldnames = ['video_id','clip_id','text', 'label', 'annotation',
            'mode', 'label_by']

    csv_header(label_csv,fieldnames)

    DATA_PATH_JSON = os.path.join(args.working_dir, 'sarcasm_data.json')
    dataset_json = json.load(open(DATA_PATH_JSON))

    if args.mode == 'genLabelDep':
        train_index_file = os.path.join(args.working_dir, 'csv/train_index0.csv')
        test_index_file = os.path.join(args.working_dir, 'csv/test_index0.csv')
    # 丁宁修改 20220315
    # independent csv our split 平衡的划分 train_5.csv
    elif args.mode == 'genLabelIndep':
        train_index_file = os.path.join(args.working_dir, 'csv/train_index5.csv')
        test_index_file = os.path.join(args.working_dir, 'csv/test_index5.csv')


    train_index = np.array(pd.read_csv(train_index_file)).reshape(-1)
    test_index = np.array(pd.read_csv(test_index_file)).reshape(-1)

    for idx, ID in enumerate(list(dataset_json.keys())[:]):
        video_id = ID
        clip_id = 0
        text = dataset_json[ID]['utterance']
        text = clean_text_remove_punctuation(text)
        # pdb.set_trace()
        label = 1.0 if dataset_json[ID]['sarcasm'] else -1.0
        annotation = 'Positive' if dataset_json[ID]['sarcasm'] else 'Negative'
        # train valid test
        mode = 'train' if idx in train_index else 'valid'
        label_by = 0
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
        video_id = ID
        clip_id = 0
        text = dataset_json[ID]['utterance']
        text = clean_text_remove_punctuation(text)
        label = 1.0 if dataset_json[ID]['sarcasm'] else -1.0
        annotation = 'Positive' if dataset_json[ID]['sarcasm'] else 'Negative'
        # train valid test
        mode = 'train' if idx in train_index else 'test'
        label_by = 0
        row = {'video_id':video_id,
                'clip_id':clip_id,
                'text':text,
                'label':label,
                'annotation':annotation,
                'mode':mode,
                'label_by':label_by,
                }

        if mode == 'test' : 
            csv_add_one_row(label_csv, fieldnames, row)
    print(f'saved in {label_csv}')
    exit()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'genLabelIndep':
        gen_label_csv(args)

    # 0315 修改 genLabelDep
    if args.mode == 'genLabelDep':
        gen_label_csv(args)
    dp = MDataPre(args)
    dp.run()
