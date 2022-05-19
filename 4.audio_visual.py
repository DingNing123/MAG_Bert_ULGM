'''
2022年 05月 11日 星期三 11:15:04 CST
'''

import librosa
import matplotlib.pyplot as plt
# 1_60_c.wav  1_60.wav 
# 1_427 1_8042 1_12002 1_10829
work_path = '/media/dn/newdisk/datasets/mmsd_raw_data/Self-MM-Processed/1_10829/tmp.wav'
y,sr = librosa.load(work_path)
# y,sr = librosa.load(r"1_60.wav")
print(y.shape)
plt.plot(y)
plt.show()
