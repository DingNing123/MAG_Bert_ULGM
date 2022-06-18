import os
import argparse

from utils.functions import Storage
from globalstr import root_dataset_dir

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'self_mm': self.__SELF_MM,
            'svm': self.__SVM,
            'mult': self.__MULT,
            'mag_bert': self.__MAG_BERT,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]

        # 丁宁修改 20220314 21:27
        # 修改特征路径 根据命令行参数的配置
        # 如果是讲话者依赖的设置 含有上下文的特征路径修改
        if args.ablationType == 'dependent_context':
            dataArgs['unaligned']['dataPath'] = '/media/dn/newdisk/datasets/Self-MM/mmsd_raw_data/Self-MM-Processed/featuresContext.pkl'

        # 0317 丁宁 修改 indep特征路径 
        elif args.ablationType == 'indep':
            dataArgs['unaligned']['dataPath'] = '/media/dn/newdisk/datasets/Self-MM/mmsd_raw_data/Self-MM-Processed/featuresIndep.pkl'

        # 丁宁 0320 修改 KeyEval 为精确度 
        if args.key_eval == 'Loss':
            pass
        elif args.key_eval == 'precision':
            dataArgs['unaligned']['KeyEval'] = 'precision'

        # dingning 0326 mode resnet152
        if args.ablationType == 'indep_resnet' :
            dataArgs['unaligned']['dataPath'] = '/media/dn/newdisk/datasets/mmsd_raw_data/Self-MM-Processed/featuresIndepResnet152.pkl'
            # dingning 0326 update visual feature dim 709 -> 2048 
            dataArgs['unaligned']['feature_dims'] = (768,33,2048)

        if args.ablationType == 'indep_context' :
            dataArgs['unaligned']['dataPath'] = '/media/dn/newdisk/datasets/mmsd_raw_data/Self-MM-Processed/featuresContextIndep.pkl'
            # dingning 0326 update visual feature dim 709 -> 2048 
            dataArgs['unaligned']['feature_dims'] = (768,33,709)


        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        # root_dataset_dir = '/media/dn/newdisk/datasets/Self-MM/'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    # 'dataPath': os.path.join(root_dataset_dir, 'Processed/features.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            },
            'mustard':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'mmsd_raw_data/Self-MM-Processed/features.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    # 这里是固定指定的 我觉得应该动态读取 
                    # 在代码 data/load_data.py(64)__init_mustard()
                    'seq_lens': (51, 615, 118), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    # here code is need update automatically 
                    'train_samples': 552 ,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp

    def __MAG_BERT(self):
        tmp = {
            'commonParas':{
            'need_data_aligned': False,
            'need_model_aligned': True,
            'need_normalized': False,
            'use_bert': True,
            'use_finetune': True,
            'save_labels': False,
            'early_stop': 20,
            # 'early_stop': 8,
            'update_epochs': 1
            },
            'datasetParas':{
                'mustard': {
                    # the batch_size of each epoch is update_epochs * batch_size
                    # 上下文特征太大，调小batch_size = 8
                    'batch_size': 8,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 5e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    # 32->64 video_out for vision:2048
                    'video_out': 64, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    # 'post_fusion_dim': 64,
                    # 768 为了特征可视化修改该参数
                    'post_fusion_dim': 768,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    # 32->64 post_video_dim for vision:2048 0403
                    'post_video_dim': 64,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 1.0
                },
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 3.0
                },
            },
        }
        return tmp

    def __MULT(self):
        tmp = {
            'commonParas':{
            'need_data_aligned': False,
            'need_model_aligned': False,
            'need_normalized': False,
            'use_bert': True,
            'use_finetune': True,
            'save_labels': False,
            'early_stop': 8,
            'update_epochs': 1
            },
            'datasetParas':{
                'mustard': {
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 5e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    # 32->64 video_out for vision:2048
                    'video_out': 64, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 64,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    # 32->64 post_video_dim for vision:2048 0403
                    'post_video_dim': 64,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 1.0
                }
            },
        }
        return tmp

    def __SVM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
            },
            'datasetParas':{
                'mustard': {
                    'batch_size': 600,
                },
                'mosi': {
                    'batch_size': 1284,
                    # train:1284 valid:229 test:686
                }
            },
        }
        return tmp

    def __SELF_MM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                # False -> True for early_fusion LSTM  0412 
                # again change to False, alignment in the model inside 0412  
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                # 0403 for mustard dataset 690 is small increase early_stop 8-> 16 8 , 12 ,16
                # for keyval from precision to Loss ,early stop adjust to 8 0412 
                'early_stop': 8,
                'update_epochs': 1
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.01,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 5e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.0,
                    # res
                    'H': 1.0
                },
                'mustard':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'batch_size': 8,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 5e-3,
                    'learning_rate_video': 5e-3,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.01,
                    'weight_decay_other': 0.001,
                    # feature subNets
                    'a_lstm_hidden_size': 16,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    # 32->64 video_out for vision:2048
                    'video_out': 64, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 64,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    # 32->64 post_video_dim for vision:2048 0403
                    'post_video_dim': 64,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args
