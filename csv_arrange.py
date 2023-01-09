import numpy as np
import pandas as pd
import utils as util
from tensorflow.keras.utils import to_categorical   


df_path = '/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/train.csv'
files_path = '/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages/'


df=pd.read_csv(df_path)
HE_exist = []
for i in range(11016):
    try:
        y_tmp = np.asarray(util.rgb2bool(util.read_image(files_path+df.loc[i,'HE'])).astype(float))
        HE_exist.append(y_tmp.max())
    except:
        HE_exist.append(0)

df['HE_exist'] = HE_exist
df.to_csv('/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/train.csv')'''
