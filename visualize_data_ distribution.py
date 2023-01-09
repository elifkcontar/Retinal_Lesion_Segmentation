import os
from re import S
from skimage import io
import numpy as np
import utils as util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data_distrubition_bg(df_path, y_path):
    #Read data
    seed = 0
    df = pd.read_csv(df_path)
    
    def iterate_df(df):
        #Iterate for every sample
        ma, hm, ex, bg = 0,0,0,0
        for i in range(len(df['4CH'])):
            y=np.asarray(util.read_var(y_path+df['4CH'].iloc[i]).astype(float))
            ma += (y[:,:,0]==1).sum()
            hm += (y[:,:,1]==1).sum()
            ex += (y[:,:,2]==1).sum()
            bg += (y[:,:,3]==1).sum()
        return [ma+hm+ex, bg]
    
    y=iterate_df(df)
    
    return y

def plot_data_distrubition_bg(train ,valid):
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
 
    # set height of bar
    Train = train
    Valid = valid
 
    # Set position of bar on X axis
    br1 = np.arange(len(Train))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
 
    # Make the plot
    plt.bar(br1, Train, color ='r', width = barWidth,
        label ='Train')
    plt.bar(br2, Valid, color ='g', width = barWidth,
        label ='Valid')
 
    # Adding Xticks
    plt.xlabel('Lesion Classes', fontweight ='bold', fontsize = 15)
    plt.ylabel('Number of pixels', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(Train))],
        ['Other', 'BG'])
 
    plt.legend()
    plt.show()

def get_data_distrubition(df_path, y_path):
    #Read data
    seed = 0
    df = pd.read_csv(df_path)
    
    def iterate_df(df):
        #Iterate for every sample
        ma, hm, ex = 0,0,0
        for i in range(len(df['4CH'])):
            y=np.asarray(util.read_var(y_path+df['4CH'].iloc[i]).astype(float))
            ma += (y[:,:,0]==1).sum()
            hm += (y[:,:,1]==1).sum()
            ex += (y[:,:,2]==1).sum()
        return [ma, hm, ex]
    
    y=iterate_df(df)
    
    return y

def plot_data_distrubition(train ,valid):
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
 
    # set height of bar
    Train = train
    Valid = valid
 
    # Set position of bar on X axis
    br1 = np.arange(len(Train))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
 
    # Make the plot
    plt.bar(br1, Train, color ='r', width = barWidth,
        label ='Train')
    plt.bar(br2, Valid, color ='g', width = barWidth,
        label ='Valid')
 
    # Adding Xticks
    plt.xlabel('Lesion Classes', fontweight ='bold', fontsize = 15)
    plt.ylabel('Number of pixels', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(Train))],
        ['MA', 'HE', 'EX'])
 
    plt.legend()
    plt.show()


df = '/home/vivente/Desktop/TEZ/Dataset/A. Segmentation/train.csv'
path = '/home/vivente/Desktop/TEZ/Dataset/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/0. All/'
data_train = get_data_distrubition(df, path)

df = '/home/vivente/Desktop/TEZ/Dataset/A. Segmentation/test.csv'
path = '/home/vivente/Desktop/TEZ/Dataset/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/0. All/'
data_test = get_data_distrubition(df, path)

plot_data_distrubition(data_train, data_test)

data_train = get_data_distrubition_bg(df, path)
data_test = get_data_distrubition_bg(df, path)
plot_data_distrubition_bg(data_train, data_test)