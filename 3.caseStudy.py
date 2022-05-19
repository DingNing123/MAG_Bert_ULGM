'''
case study
model_path='/media/dn/newdisk/datasets/Self-MM/results/models/mag_bert-mustard-regression.pth'
print(model_path)
'''
import os
import gc
import time
import random
import torch
import pynvml
import logging
import argparse
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_tune import ConfigTune
from config.config_regression import ConfigRegression
from globalstr import root_dataset_dir

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# -1: cpu; 0: gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1, 2, 3]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)

    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')


    atio = ATIO().getTrain(args)
    # do train
    # best_epoch = atio.do_train(model, dataloader)
    # atio.do_train(model, dataloader) 0404 before it return None
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)

    # do test
    # import ipdb;ipdb.set_trace()
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # here add results['best_epoch'] = best_epoch  0404  add best_epoch result 
    # results['best_epoch'] = best_epoch 
    return results

def run_tune(args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = [] # save used paras
    save_file_path = os.path.join(args.res_save_dir, \
                                f'{args.datasetName}-{args.modelName}-{args.train_mode}-tune.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    
    for i in range(tune_times):
        # load free-most gpus
        pynvml.nvmlInit()
        # cancel random seed
        setup_seed(int(time.time()))
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        print(args)
        # print debugging params
        logger.info("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, tune_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#"*90)
        logger.info('Start running %s...' %(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        results = []
        for j, seed in enumerate([1111]):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args))
        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))

        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' %(save_file_path))

def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        args = config.get_config()
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' %(args.modelName))
        logger.debug(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args)
        # restore results
        model_results.append(test_results)

    criterions = list(model_results[0].keys())
    # load other results
    # add date 0404 2022 
    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}-0404.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    resValues = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        # here add best_epoch 5 0404 record 5 best_epoch not mean and std
        if c == 'best_epoch':
            res.append(values)
            # cannot set a row with mismatched columns
            resValues.append(values)
        else:
            # 0404 output 5 results besides mean std
            res.append((mean, std))
            # 0406 two lines 
            values = [round(v*100,2) for v in values]
            resValues.append(values)

    df.loc[len(df)] = res
    # 0406 ding ning two lines
    df.loc[len(df)+1] = resValues 

    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))

def set_log(args):
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)

    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?')
    # 丁宁修改20220317 
    # 消融实验 1 默认的依赖 最终语句
    # 2 讲话者依赖 最终语句+ 上下文
    # 3 讲话者独立 最终语句
    # 4 讲话者独立 最终语句+ 上下文
    # 5 讲话者独立 最终语句 resnet152 
    parser.add_argument('--ablationType', type=str, default="dependent_final", help='dependent_final/dependent_context/indep/indep_context/indep_resnet')

    # ding ning modified in 0320
    # `Loss` used in regression
    # `f1_score` used in classification
    # `precision` used in classification 
    # `recall` used in classification 
    parser.add_argument('--key_eval',type=str,default="Loss",help='Loss/f1_score/precision/recall')

    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression / classification')
    parser.add_argument('--modelName', type=str, default='self_mm',
                        help='support self_mm/svm/mult/mag_bert')
    parser.add_argument('--datasetName', type=str, default='mustard',
                        help='support mosi/mosei/sims/mustard')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default = root_dataset_dir + 'results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default = root_dataset_dir + 'results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # I think there is an error .
    # if args.datasetName == 'mustard' :
        # root_dataset_dir = '/media/dn/newdisk/datasets/CMU_MOSI/Raw/'
        
    logger = set_log(args)
    # for data_name in ['sims', 'mosi', 'mosei']:
    for data_name in [ args.datasetName ]:
        args.datasetName = data_name
        # args.seeds = [1111,1112, 1113, 1114, 1115]
        # notes next line or not to change run times
        # args.seeds = [1113,1114]
        args.seeds = [1114]
        if args.is_tune:
            run_tune(args, tune_times=50)
        else:
            run_normal(args)
