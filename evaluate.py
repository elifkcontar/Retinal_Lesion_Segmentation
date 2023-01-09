import numpy as np
import pandas as pd
import time

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)
print(gpus[0])
#tf.config.run_functions_eagerly(True)
from utils import load_model, feat_train_config, make_subfolder, make_log_dir, write_to_log
from feat_train_functions import *
from model import residual_unet, fast_fcn, FastFCN

#Define paths
patched_dataset_path = 'C:/Users/Asus/Desktop/TEZ/L_Seg_Augmented_Data/'
train_df_path = patched_dataset_path + 'train.csv'
test_df_path = patched_dataset_path + 'test.csv'
x_train_path = patched_dataset_path + '1. Original Images/c. Bens Training Set/' 
y_train_path = patched_dataset_path + '2. All Segmentation Groundtruths/a. Training Set/0. All/'
x_test_path = patched_dataset_path + '1. Original Images/d. Bens Testing Set/'
y_test_path = patched_dataset_path + '2. All Segmentation Groundtruths/b. Testing Set/0. All/'
full_size_dataset_path = "C:/Users/Asus/Desktop/Codes/datasets/A. Segmentation_adems/"
full_size_df_path = full_size_dataset_path + 'test.csv'
x_full_size_path = full_size_dataset_path + '1. Original Images/d. Bens Testing Set/'
y_full_size_path = full_size_dataset_path + '2. All Segmentation Groundtruths/b. Testing Set/0. All/'

#Read dataframe
df=pd.read_csv(train_df_path)
full_size_df = pd.read_csv(full_size_df_path)
test_df=pd.read_csv(test_df_path)

#Define variables
conf = feat_train_config(name= "config", MODEL_NAME='fast_fcn', CLASS_NUM=3, IMG_SIZE=512, BATCH_SIZE=2, _1ST_STEP_EPOCH = -1, _2ND_STEP_EPOCH = 10, LOSS_WEIGHT=1)
#Set seed
np.random.seed(42)

conf.log_dir = "out/kabuska6/"
if(conf.MODEL_NAME=='fast_fcn'):
    model = FastFCN(conf.IMG_SIZE, conf.IMG_SIZE, conf.CLASS_NUM)

model.model.load_weights("out/kabuska6/weights-improvement-05.h5")
start_time = time.time()
make_subfolder("visualisations", conf.log_dir)
make_subfolder("prediction", conf.log_dir)

get_full_size_metric(model, full_size_df, x_full_size_path, y_full_size_path, conf, name='center_merge', stride = 256)
#get_full_size_metric(model, full_size_df, x_full_size_path, y_full_size_path, conf, name='normal_merge', stride = conf.IMG_SIZE)

elapsed_time = time.time() - start_time
evaluate_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
write_to_log(conf.log_dir, "\n\nElapsed time in the evaluation: " + evaluate_time)