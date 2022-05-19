'''
2022年 05月 14日 星期六 11:34:47 CST
'''
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
# from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import os

#M_FUSION_DATA_PATH = '/media/dn/home/dn/4.datasets/m_fusion_data/'
#modal = 'audio'
#path =  M_FUSION_DATA_PATH  + f'representation/{modal}.npz'
path='mult.npz'
# path='magbert.npz'
# path='oursv1.npz'
# path='oursv2.npz'
print(path)
data = np.load(path)
print(data.files)
# import ipdb;ipdb.set_trace()

X = data['repre']
y4 = data['label']
y4 = [1 if yi==1.0 else 0 for yi in y4]
y4 = np.array(y4)
#ipdb> X.shape
#(552, 16)
#ipdb> y4.shape
#(552,)

# non-sarcstic :N sarcastic: Y
class_name=['N','Y']
y4 = [class_name[yi] for yi in y4]

tsne = TSNE()
palette = sns.color_palette("bright",2)
X_embedded = tsne.fit_transform(X)
#ipdb> X_embedded.shape
#(552, 2)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y4, legend='full', palette=palette)
plt.title('mult',fontsize=10)
plt.legend(fontsize=10)
plt.show()

'''
import numpy as np
a=np.arange(3)
b=np.arange(4)
c=np.arange(5)
np.savez('array_save.npz',a,b,c_array=c)

import numpy as np
A=np.load('array_save.npz')
print(A['arr_0'])
print(A['arr_1'])
print(A['c_array'])
'''
