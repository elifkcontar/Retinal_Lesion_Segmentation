import numpy as np
import pandas as pd
import time

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras import backend as K
import utils as util
from utils import feat_train_config, make_subfolder, make_log_dir, show_images, write_to_log, save_var, save_model, read_var
from model import residual_unet, fast_fcn, FastFCN
from feat_train_functions import generator, plot_loss_and_acc, get_full_size_metric, get_full_size_metric_binary, plot_loss_and_acc_train

from tensorflow.keras.optimizers import Adam
from model import ensembled_loss
#Define paths
patched_dataset_path = '/content/dataset/'
train_df_path = patched_dataset_path + 'train.csv'
test_df_path = patched_dataset_path + 'test.csv'
x_train_path = patched_dataset_path + '1. Original Images/e. Adems Training Set/' 
y_train_path = patched_dataset_path + '2. All Segmentation Groundtruths/a. Training Set/0. All/'
x_test_path = patched_dataset_path + '1. Original Images/f. Adems Testing Set/'
y_test_path = patched_dataset_path + '2. All Segmentation Groundtruths/b. Testing Set/0. All/'
full_size_dataset_path = "not defined/"
full_size_df_path = full_size_dataset_path + 'test.csv'
x_full_size_path = full_size_dataset_path + '1. Original Images/d. Bens Testing Set/'
y_full_size_path = full_size_dataset_path + '2. All Segmentation Groundtruths/b. Testing Set/0. All/'

#Read dataframe
df=pd.read_csv(train_df_path)
#full_size_df = pd.read_csv(full_size_df_path)
test_df=pd.read_csv(test_df_path)

#Define variables
candidate_config_list = []
config_1 = feat_train_config(name= "config", MODEL_NAME='fast_fcn', CLASS_NUM=3, IMG_SIZE=512, BATCH_SIZE=4, _1ST_STEP_EPOCH = -1, _2ND_STEP_EPOCH = 10, LOSS_WEIGHT=1)
#config_2 = feat_train_config(name= "config", MODEL_NAME='residual_unet', CLASS_NUM=4, BATCH_SIZE=2, _1ST_STEP_EPOCH = 3, _2ND_STEP_EPOCH = 15, LOSS_WEIGHT=0.5)
candidate_config_list.append(config_1)
#candidate_config_list.append(config_2)


for conf in candidate_config_list:
    tf.keras.backend.clear_session()
    #Set seed
    np.random.seed(42)
    print(conf.name)
    log_dir = make_log_dir('/content/drive/MyDrive/TezColab/codes/out')
    write_to_log(log_dir, "train_df_path: \n" + train_df_path)
    write_to_log(log_dir, "\n\ntest_df_path: \n" + test_df_path)
    write_to_log(log_dir, "\n\nx_train_path: \n" + x_train_path)
    write_to_log(log_dir, "\n\ny_train_path: \n" + y_train_path)
    write_to_log(log_dir, "\n\nx_test_path: \n" + x_test_path)
    write_to_log(log_dir, "\n\ny_test_path: \n" + y_test_path)
    conf.save(save_dir = log_dir)

    if(conf.MODEL_NAME=='fast_fcn'):
        model = FastFCN(conf.IMG_SIZE, conf.IMG_SIZE, conf.CLASS_NUM)
    elif(conf.MODEL_NAME=='residual_unet'):
        model = residual_unet(input_size=(conf.IMG_SIZE, conf.IMG_SIZE,3), out_size=conf.CLASS_NUM)
    else:
        ('Please provide a valid model name')
        break

    model.model.load_weights('/content/drive/MyDrive/TezColab/weights-improvement-02.h5')
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    weights_dir = make_subfolder("weights",log_dir)
    my_filepath = weights_dir + 'weights-improvement-{epoch:02d}.h5'
    
    callback2 = tf.keras.callbacks.ModelCheckpoint(
        filepath=my_filepath,
        save_freq='epoch'
    )
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
    #                          patience=4, min_lr=0.000005)

    train_df = df[df['Background'] == False]   #For multiclass
    #train_df = df[df['MA_exist'] == 1]    #For binary
    train_df.reset_index(drop=True, inplace=True)

    #Create generators
    train_gen = generator(train_df, x_train_path, y_train_path, config= conf, augmentation=True, mode='multi_pos')
    test_gen = generator(test_df, x_test_path, y_test_path, config= conf, augmentation=False, mode='multi_pos')
    #train_gen = generator(train_df, x_train_path, y_train_path, config= conf, augmentation=True, mode='binary')
    #test_gen = generator(test_df, x_test_path, y_test_path, config= conf, augmentation=False, mode='binary')

    start_time = time.time()
    
    #Train 1st step
    '''
    model.set_trainable(False)
    model.compile()
    model.model.fit(train_gen,
            epochs=conf._1ST_STEP_EPOCH,
            steps_per_epoch=len(train_df.index) // (conf.BATCH_SIZE),
            validation_data=test_gen,
            validation_steps= 1,#len(test_df.index) // (conf.BATCH_SIZE),
            callbacks=[callback2])
    
    elapsed_time = time.time() - start_time
    train_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    write_to_log(log_dir, "\n\nElapsed time in the pre-train: " + train_time)
    start_time = time.time()
    '''
    
    #Train 2nd step
    model.set_trainable(True)
    opt = Adam(learning_rate=1e-8)
    model.model.compile(loss=ensembled_loss,
            optimizer = opt,
            run_eagerly=True)
    history=model.model.fit(train_gen,
                    epochs=conf._2ND_STEP_EPOCH,
                    steps_per_epoch=len(train_df.index) // (conf.BATCH_SIZE),
                    validation_data=test_gen,
                    validation_steps= 1,#len(test_df.index) // (conf.BATCH_SIZE)
                    callbacks=[callback2]
                    ).history

    elapsed_time = time.time() - start_time
    train_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    write_to_log(log_dir, "\n\nElapsed time in the train: " + train_time)
    save_var(history, log_dir + "history")
    #Save model and weights
    save_model(model.model, log_dir + conf.MODEL_NAME +  '_trained_model')

    save_dir = make_subfolder("visualisations/", conf.log_dir)
    save_dir = make_subfolder("prediction/", conf.log_dir)

    #get_full_size_metric(model.model, full_size_df, x_full_size_path, y_full_size_path, conf, name='center_merge', stride = 256)
    #get_full_size_metric(model, full_size_df, x_full_size_path, y_full_size_path, conf, name='normal_merge', stride = conf.IMG_SIZE)
    
    #Visualize history and metrics
    plot_loss_and_acc_train(history, save=True, saveDir=log_dir)

