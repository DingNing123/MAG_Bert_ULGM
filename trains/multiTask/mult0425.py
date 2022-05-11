'''
trains/multiTask/mult.py
ding ning 
2022年 04月 22日 星期五 17:39:49 CST
'''
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger('MSA.trains.mult')

class MULT():
    def __init__(self,args):
        pass

    def do_train(self,model,dataloader):
        with tqdm(dataloader['train']) as td:
            for batch_data in td:
                bd = batch_data

                device = torch.device('cuda:0')
                vision = bd['vision'].to(device)
                # vision.shape torch.Size([10, 46, 2048])
                audio = bd['audio'].to(device)
                # ipdb> audio.shape torch.Size([10, 645, 33])
                text= bd['text'].to(device)
                # text.shape torch.Size([10, 45, 768])
                audio_lengths = bd['audio_lengths'].to(device)
                # audio_lengths tensor([150, 176,  69, 130, 148, 178, 260, 592,  22, 188])
                vision_lengths = bd['vision_lengths'].to(device)
                # vision_lengths tensor([12, 14,  6, 11, 12, 14, 20, 43,  3, 15])
                outputs = model(text,(audio,audio_lengths),(vision,vision_lengths))
                import ipdb;ipdb.set_trace()

                pass




    def do_test(self,model,dataloader,mode="VAL"):
        pass


