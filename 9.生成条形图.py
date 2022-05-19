'''
bar.npz
生成条形图
'''

import matplotlib.pyplot as plt
import numpy as np

npzfile = np.load('bar.npz')
# ipdb> npzfile.files ['y_true', 'y_pred', 'id']
y_true = npzfile['y_true']
y_pred = npzfile['y_pred']
ID = npzfile['id']
error_index=[]
for i,val in enumerate(y_true):
    if val != y_pred[i]:
        error_index.append(i)
print(error_index)
# import ipdb;ipdb.set_trace()
ID = np.array(ID)
print(ID)
error_id_list = ID[error_index]
error_id_list = [one.split('$')[0] for one in  error_id_list ]
import json
import os
DATA_PATH_JSON = os.path.join('/media/dn/newdisk/datasets/mmsd_raw_data/', 'sarcasm_data.json')
dataset_json = json.load(open(DATA_PATH_JSON))
x,y,z,w=0,0,0,0
for err_id in error_id_list:
    if dataset_json[err_id]['speaker']=='SHELDON' and dataset_json[err_id]['sarcasm']==True:
        x += 1
    if dataset_json[err_id]['speaker']=='SHELDON' and dataset_json[err_id]['sarcasm']==False:
        y += 1
    if dataset_json[err_id]['speaker']=='HOWARD' and dataset_json[err_id]['sarcasm']==True:
        z += 1
    if dataset_json[err_id]['speaker']=='HOWARD' and dataset_json[err_id]['sarcasm']==False:
        w += 1
# 13 22 9 10 54
print(x,y,z,w,sum([x,y,z,w]))
# 堆叠柱状图
# https://zhuanlan.zhihu.com/p/25128216
import numpy as np
import matplotlib.pyplot as plt

size = 2
x_ax = ['SHELDON','HOWARD']
# x = np.arange(size)
a = [y,w]
b = [x,z]
# b = np.random.random(size)
print(a,b)
# [0.13676426 0.07171267] [0.0850228  0.60664246]

plt.bar(x_ax, a, label='-1')
plt.bar(x_ax, b, bottom=a, label='1')
plt.legend()
plt.ylabel('number')
plt.xlabel('speaker')
plt.title('error predict')
plt.show()
