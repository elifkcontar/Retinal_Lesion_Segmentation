import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage as sk
from skimage.transform import resize
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from tensorflow.keras.utils import to_categorical   
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import utils as util
from utils import load_model, rgb2gray, rgb2bool, feat_train_config, make_subfolder, make_log_dir, write_to_log, save_var, save_model
from feat_train_functions import *

from JPU import JPU_DeepLab

def plot_roc_curve(y_true, y_pred,
                    title='ROC Curve',
                    save=True,
                    saveDir='out/'):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    #Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label=f"AUC = {roc_auc:0.2f}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    if(save):
        plt.savefig(saveDir + title+ '.png')

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          saveDir='out/'):
    """
    This function calculates, prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, digits=3)
    print(report)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if(save):
        plt.savefig(saveDir + title+ '.png')
    #plt.show(block=False)
    return report

#Define paths
patched_dataset_path = '/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/'
train_df_path = patched_dataset_path + 'train.csv'
test_df_path = patched_dataset_path + 'test.csv'
x_train_path = patched_dataset_path + '1. Original Images/c. Bens Training Set/'
x_test_path = patched_dataset_path + '1. Original Images/d. Bens Testing Set/'

#Read dataframe
train_df=pd.read_csv(train_df_path)
test_df=pd.read_csv(test_df_path)
train_df['Background'] = train_df['Background'].astype('str')
test_df['Background'] = test_df['Background'].astype('str')
#Define variables
candidate_config_list = []
config_4 = feat_train_config(name= "config", MODEL_NAME='vgg16', IMG_SIZE=512, CLASS_NUM=4, _1ST_STEP_EPOCH = 0, _2ND_STEP_EPOCH = 30, BATCH_SIZE= 8)
candidate_config_list.append(config_4)

for conf in candidate_config_list:
    #Set seed
    seed = np.random.seed(42)
    print(conf.name)
    log_dir = make_log_dir('out_classification/')
    write_to_log(log_dir, "train_df_path: \n" + train_df_path)
    write_to_log(log_dir, "\n\ntest_df_path: \n" + test_df_path)
    write_to_log(log_dir, "\n\nx_train_path: \n" + x_train_path)
    write_to_log(log_dir, "\n\nx_test_path: \n" + x_test_path)
    conf.save(save_dir = log_dir)


    #Create generators
    train_datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    brightness_range=[0.8, 1],
                    shear_range=0,
                    zoom_range=[0.8, 0.85],
                    channel_shift_range=0,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode = "constant",
                    rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe   =train_df,
            directory   =x_train_path,
            x_col       ='Input',
            y_col       ='Background',
            target_size =(conf.IMG_SIZE, conf.IMG_SIZE),
            class_mode  ='binary',
            batch_size  =conf.BATCH_SIZE,
            seed        =seed,
            shuffle     =True)

    test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=x_test_path,
            x_col="Input",
            y_col="Background",
            target_size=(conf.IMG_SIZE, conf.IMG_SIZE),
            class_mode='binary',
            batch_size=conf.BATCH_SIZE,
            seed=seed,
            shuffle=False)
    
    #MODEL
    '''
    inputs = Input(shape=(conf.IMG_SIZE, conf.IMG_SIZE, 3))
    effnet = VGG16(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    l=GlobalAveragePooling2D()(effnet.output)
    l=Dense(128,activation='relu')(l)
    l=Dropout(0.4)(l)
    out=Dense(1, activation='sigmoid')(l)
    model = Model(inputs=effnet.input, outputs=out)
    '''
    model = load_model('/home/vivente/Desktop/TEZ/Code/segmentation/out_classification/2021_November_27-15_57_51/fcn_trained_model')
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.optimizers.Adam(learning_rate=5e-5),
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    weights_dir = make_subfolder("weights",log_dir)
    my_filepath = weights_dir + 'weights-improvement-{epoch:02d}.hdf5'

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.000005)
    #Train df 2nd step
    start_time = time.time()
    
    history=model.fit(train_generator,
                    epochs=conf._2ND_STEP_EPOCH,
                    steps_per_epoch=len(train_df.index) // (conf.BATCH_SIZE),
                    validation_data=test_generator,
                    validation_steps= len(test_df.index) // (conf.BATCH_SIZE),
                    callbacks=[callback, reduce_lr]).history
    
    elapsed_time = time.time() - start_time
    train_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    write_to_log(log_dir, "\n\nElapsed time in the train: " + train_time)
    save_var(history, log_dir + "history")
    #Save model and weights
    save_model(model, log_dir + conf.MODEL_NAME +  '_trained_model')

    #Visualize history and metrics
    
    plot_loss_and_acc(history, save=True, saveDir=log_dir)
    scores = model.evaluate(test_generator)
    write_to_log(log_dir, "\nTest Loss Pretrained:{}".format(scores[0]))
    write_to_log(log_dir, "\nTest Accuracy Pretrained:{}".format(scores[1]))
    print("Test Loss Pretrained:{}".format(scores[0]))
    print("Test Accuracy Pretrained:{}".format(scores[1]))   
    

    y_pred = model.predict(test_generator)
    test_df=pd.read_csv(test_df_path)
    y_true = test_df['Background'].astype('uint8')
    #ROC Curve and Area
    plot_roc_curve(y_true, y_pred, save=True, saveDir=log_dir)
    #Save Confusion matrix
    y_pred = y_pred > 0.5
    report = plot_confusion_matrix(y_true, y_pred, classes=[0, 1], save=True, saveDir=log_dir, normalize=True, title='Confusion matrix percentage')
    report = plot_confusion_matrix(y_true, y_pred, classes=[0, 1], save=True, saveDir=log_dir, normalize=False)
    #util.send_as_mail(log_dir)
