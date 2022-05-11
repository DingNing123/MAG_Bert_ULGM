import ipdb
from sklearn.metrics import classification_report
import os
import pandas as pd
import logging
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger('MSA')

class SVM():
    def __init__(self, args):
        self.args = args 
        pass

    def do_train(self, model, dataloader):
        for batch_data in dataloader['train']:
            bd = batch_data
        # load all data batch_size = 600 > 554 
        y = bd['labels']['M'].view(-1)
        import numpy as np
        y = np.array(y)
        # 使用Bert模型第一个token表示整个句子的含义
        text = np.array(bd['text'][:,0,:])
        # audio和vision没有考虑每个序列的不同长度，这是填充后求的平均，对结果可能有影响。
        audio = np.mean(np.array(bd['audio']),axis=1)
        vision = np.mean(np.array(bd['vision']),axis=1)
        X = np.concatenate([text,audio,vision],axis=1)
        clf = make_pipeline(StandardScaler(), SVC(C=10,gamma='scale',kernel='rbf'))
        # import ipdb;ipdb.set_trace()
        # 对于mosi数据集来说，需要把y转化为离散的int类型。
        y = [1 if one >=0 else -1 for one in y]
        y = np.array(y)

        clf.fit(X, y)
        # start test 
        for batch_data in dataloader['test']:
            bd = batch_data
        y_true = bd['labels']['M'].view(-1)
        y_true = np.array(y_true)

        text = np.array(bd['text'][:,0,:])
        audio = np.mean(np.array(bd['audio']),axis=1)
        vision = np.mean(np.array(bd['vision']),axis=1)
        X = np.concatenate([text,audio,vision],axis=1)
        y_pred = clf.predict(X)

        # import ipdb;ipdb.set_trace()
        y_true = [1 if one >=0 else -1 for one in y_true]
        y_true = np.array(y_true)
        
        result_dict = classification_report(y_true,y_pred,output_dict=True)
        recall = result_dict["weighted avg"]["recall"]
        recall = round(recall*100,2)
        precision = result_dict["weighted avg"]["precision"]
        precision = round(precision*100,2)
        f1_score = result_dict["weighted avg"]["f1-score"]
        f1_score = round(f1_score*100,2)

        args = self.args
        save_path = os.path.join(args.res_save_dir, f'{args.datasetName}-{args.train_mode}-0404.csv')

        logger.info(save_path)

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=['Model','precision','recall','f1_score','Loss','best_epoch'])

        res = [args.modelName]
        res.append(precision)
        res.append(recall)
        res.append(f1_score)
        res.append('-')
        res.append('-')
        df.loc[len(df)] = res 
        df.to_csv(save_path, index=None)
        # 2022/04/21
        logger.info('precision,recall,f1_score')
        logger.info('%s,%s,%s'%(precision,recall,f1_score))

        logger.info("Results are saved to %s ." % (save_path))
        exit()

        # ipdb.set_trace(context = 5)


    def do_test(self, model, dataloader,mode="VAL"):
        pass
    

