#!/bin/bash
# 2022年 05月 15日 星期日 12:11:19 CST
# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mult
# python 6.生成tsne的输入文件npz.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mult
# python 5.特征可视化tsne.py


# 2022年 05月 15日 星期日 11:48:37 CST 这三个是一组实验，需要按顺序运行完才可以生成可视化特征

# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mag_bert
# python 6.生成tsne的输入文件npz.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mag_bert
# python 5.特征可视化tsne.py

# 2022年 05月 13日 星期五 13:57:18 CST
# python 3.caseStudy.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mag_bert
# 2022年 05月 14日 星期六 12:30:41 CST
# python 6.生成tsne的输入文件npz.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mag_bert
# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName self_mm
# 2022年 05月 13日 星期五 17:31:51 CST
# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mag_bert

# 2022年 05月 15日 星期日 10:22:42 CST 这三个是一组实验，需要按顺序运行完才可以生成可视化特征
# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName self_mm
# python 6.生成tsne的输入文件npz.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName self_mm
# python 5.特征可视化tsne.py




# python 4.audio_visual.py
# 2022年 05月 14日 星期六 11:43:37 CST
# python 5.特征可视化tsne.py
# python run.py --datasetName mustard --ablationType indep_context --key_eval Loss --modelName self_mm
# 2022年 05月 09日 星期一 18:13:58 CST
# python run.py --datasetName mustard --ablationType indep_context --key_eval Loss --modelName mag_bert
# python run.py --datasetName mustard --ablationType dependent_context --key_eval Loss --modelName mag_bert
# python run.py --datasetName mosi --key_eval Loss --modelName mag_bert
# python run.py --datasetName mosi --key_eval Loss --modelName svm
# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName mult
# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss --modelName svm

# 生成独立设置特征 
# python data/DataPreForMustard.py  --mode runIndep
# python  -m pdb data/DataPreForMustard.py  --mode runIndep
# 0322 丁宁修改 集成Resnet152 2048 dim feature
# python  -m pdb data/DataPreForMustard.py  --mode runIndepResnet152
# 使用所有数据进行测试 生成讲话者独立的设置 03152022   
# python  data/DataPreForMustard.py  --mode genLabelDep
# python  data/DataPreForMustard.py  --mode genLabelIndep
#python  -m pdb data/DataPreForMustard.py  --mode genLabelIndep

# 运行mustarad数据集, 讲话者独立  最终语句 0317
# python  run.py --datasetName mustard --ablationType indep
# python -m pdb run.py --datasetName mustard --ablationType indep
# 0320 add KeyEval = Acc 
# python -m pdb run.py --datasetName mustard --ablationType indep --key_eval precision 
# 讲话者独立 最终语句 resnet152 
# python run.py --datasetName mustard --ablationType indep_resnet --key_eval Loss

# 运行mustarad数据集, 讲话者依赖 最终语句+ 上下文
# python run.py --datasetName mustard --ablationType dependent_context

# debug 运行mustarad数据集, 讲话者依赖 最终语句+ 上下文
# python -m pdb run.py --datasetName mustard --ablationType dependent_context
# 运行mustarad数据集
# python run.py --datasetName mustard
# 调试mustarad数据集
# python -m pdb run.py --datasetName mustard

# 使用所有数据进行测试
# python -m pdb  data/DataPreForMustard.py  --mode run
# 0322 丁宁 尝试集成ResNet152 特征 dependent 暂时去实现independent 
# python -m pdb  data/DataPreForMustard.py  --mode run
# 使用20条数据进行测试
# python -m pdb  data/DataPreForMustard.py  --mode debug

# 使用20条数据进行测试Contextual
# python -m pdb data/DataPreForMustardWithContext.py --mode debug
# python -m pdb data/2.DataPreForMustardWithContext.py --mode debug

# 生成讲话者独立 csv 
# python data/3.DataPreForMustardWithContextSpeakerIndependentBanlanced.py --mode run
# 上下文的消融实验是在709维的特征下做的。
