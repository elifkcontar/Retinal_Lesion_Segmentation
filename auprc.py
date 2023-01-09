import os
import numpy as np
from sklearn.metrics import auc, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time
import csv
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical   
import skimage.io as io
from itertools import cycle

def read_var(file_name):   
    infile = open(file_name,'rb')
    var = pickle.load(infile)
    infile.close()
    return var 

def load_model(name):
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+'.h5')
    print("Loaded model from disk")
    return loaded_model

def save_var(var, file_name):
    '''
    Saves any type of variable with the given filename(can be a path)
    '''
    out_file = open(file_name,'wb')
    pickle.dump(var,out_file)
    out_file.close()

def precision_recall(y_true, y_pred):
    prec, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return prec, recall

def patched_predict(img, model, window_size = 512, stride = 256, channel = 5):
    bound = (window_size - stride)//2
    #print(img.shape[0],img.shape[1],img.shape[2])
    #fullsize_pred = np.zeros((img.shape[0], img.shape[1], channel))
    tmp_arr = np.full((img.shape[0], img.shape[1]), channel-1)
    fullsize_pred = y_pred=to_categorical(tmp_arr, num_classes=channel)

    row=(img.shape[0] - 2*bound)//stride
    column=(img.shape[1] - 2*bound)//stride
    model_class = load_model('/home/vivente/Desktop/TEZ/Code/out_classification/2021_November_27-15_57_51/fcn_trained_model')
    #model_class = 0
    background_array = np.full((window_size, window_size), 3)
    background_array = np.squeeze(np.eye(4)[background_array.reshape(-1)])
    for i in range(row):
        for j in range(column):
            #print(i,j)
            x_start = (i*stride)
            x_stop = x_start + window_size
            y_start = (j*stride)
            y_stop = y_start + window_size
            bx_start = x_start + bound
            bx_stop  = x_stop - bound
            by_start = y_start + bound
            by_stop  = y_stop - bound
            #print(x_start,x_stop,y_start,y_stop)
            #print(bx_start,bx_stop,by_start,by_stop)
            pred_in = img[x_start:x_stop, y_start:y_stop, :]
            #part_pred_class = model_class.predict(np.reshape(pred_in, (1,window_size, window_size, 3)))
            part_pred_class = 0
            if(part_pred_class<0.5):
                part_pred = model.predict(np.reshape(pred_in, (1,window_size, window_size, 3)))
            else:
                part_pred = background_array
            part_pred = np.reshape(part_pred, (window_size, window_size, channel))
            #print(np.shape(part_pred))
            ################ Koseler ###############
            if(i==0 and j== 0):                     #####baslangic kösesi
                valid_part = part_pred[0:window_size - bound, 0:window_size - bound, :]
                #print(np.shape(valid_part))
                fullsize_pred[0:bx_stop, 0:by_stop, :] = valid_part
            elif(i==0 and j== column-1):            #####sag kose
                valid_part = part_pred[0:window_size - bound, bound:window_size, :]
                #print(np.shape(valid_part))
                fullsize_pred[0:bx_stop, by_start:y_stop, :] = valid_part
            elif(i==row-1 and j==column-1):         ####sag alt kose
                valid_part = part_pred[bound:window_size, bound:window_size,:]
                #print(np.shape(valid_part))
                fullsize_pred[bx_start:x_stop, by_start:y_stop, :] = valid_part
            elif(i==row-1 and j==0):                ####sol alt kose
                valid_part = part_pred[bound:window_size, 0:window_size - bound,:]
                #print(np.shape(valid_part))
                fullsize_pred[bx_start:x_stop, y_start:by_stop, :] = valid_part
            ################ Kenarlar ##############
            elif(i==0 and j!=0 and j!= column-1):   #####ust kenar
                valid_part = part_pred[0:window_size - bound, bound:window_size - bound, :]
                #print(np.shape(valid_part))
                fullsize_pred[0:bx_stop, by_start:by_stop, :] = valid_part
            elif(j==0 and i!=0 and i!=row-1):       #####sol kenar
                valid_part = part_pred[bound:window_size - bound, 0:window_size - bound, :]
                #print(np.shape(valid_part))
                fullsize_pred[bx_start:bx_stop, 0:by_stop, :] = valid_part
            elif(i==row-1 and j!=column-1 and j!=0):####alt kenar
                valid_part = part_pred[bound:window_size, bound:window_size - bound, :]
                #print(np.shape(valid_part))
                fullsize_pred[bx_start:x_stop, by_start:by_stop, :] = valid_part
            elif(j==column-1 and i!=row-1 and i!=0):#####sag kenar
                valid_part = part_pred[bound:window_size - bound, bound:window_size,:]
                #print(np.shape(valid_part))
                fullsize_pred[bx_start:bx_stop, by_start:y_stop, :] = valid_part

            else:
                valid_part = part_pred[bound:window_size - bound, bound:window_size - bound,:]
                #print(np.shape(valid_part))
                fullsize_pred[bx_start:bx_stop, by_start:by_stop, :] = valid_part
    '''
    print("Full size prediction completed")
    colour_img_pred = util.colorize_binary_img(fullsize_pred)
    plt.imshow(colour_img_pred)
    plt.show()
    '''

    return np.asarray(fullsize_pred)

def plot_pr_curve(precision, recall, n_classes):
    if(n_classes==1):
        plt.plot(recall, precision, lw=2 )
    else:
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
        
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

def calculate_auc_pr_curve(fname, model, test_df, x_test_path, y_test_path, stride = 256):
    IMG_SIZE = 512
    IMG=io.imread(x_test_path+test_df.loc[0,'Input'])
    IMG_SIZE_0 = IMG.shape[0]
    IMG_SIZE_1 = IMG.shape[1]
    CLASS_NUM = 4
    aupr_list = []
    test_labels = []
    test_preds = []
    sample_num = 27

    for i in range(sample_num):
        
        
        x=np.asarray(io.imread(x_test_path+test_df.loc[i,'Input']))
        x=x/255.
        y=np.asarray(read_var(y_test_path+test_df.loc[i,'4CH']).astype(float))

        pred = (patched_predict(x, model, window_size = IMG_SIZE, stride = stride, channel = CLASS_NUM))

        test_labels.append(np.asarray(y).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM))
        test_preds.append(np.asarray(pred).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM))
        


        #*******************DELETE AFTER ONE TIME***********************************
        save_var(pred, fname + 'prediction/'+'prob__y_pred_'+str(i+55))
        #***************************************************************************
        #test_preds.append(np.asarray(read_var(fname + 'prediction/'+'prob__y_pred_'+str(i+55))).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM))
        #test_labels.append(np.asarray(read_var('/home/vivente/Desktop/TEZ/Code/segmentation/out/2021_December_11-18_00_35/' + 'prediction/'+'y_true_'+str(i+55))).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM))

    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    _, ax = plt.subplots(figsize=(7, 8))
    test_labels = np.asarray(test_labels).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
    test_preds = np.asarray(test_preds).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)

    for j, color in zip(range(CLASS_NUM-1), colors): 
    #for j in range(CLASS_NUM-1):    
        prec, recall, thresholds = precision_recall_curve(test_labels[:,j], test_preds[:,j])
        aupr = auc(recall, prec)
        print(aupr)
        aupr_list.append(aupr)
        display = PrecisionRecallDisplay.from_predictions(test_labels[:,j], test_preds[:,j])
        display.plot(ax=ax, name=f"Precision-recall for class {j}", color=color)

    # set the legend and the axes
    handles, labels = display.ax_.get_legend_handles_labels()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")
    plt.show()

    filename = "auprc.csv"
    fields = ['MA', 'HE', 'EX']
    with open(fname + filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerow(aupr_list)

"""
fname = '/home/vivente/Desktop/TEZ/Code/segmentation/out/2022_January_13-11_11_51/'
full_size_dataset_path = '/home/vivente/Desktop/TEZ/Dataset/A. Segmentation/'
full_size_df_path = full_size_dataset_path + 'test.csv'
x_full_size_path = full_size_dataset_path + '1. Original Images/d. Bens Testing Set/'
y_full_size_path = full_size_dataset_path + '2. All Segmentation Groundtruths/b. Testing Set/0. All/'
full_size_df = pd.read_csv(full_size_df_path)
model = load_model(fname+'residual_unet_trained_model')
calculate_auc_pr_curve(fname, model, full_size_df, x_full_size_path, y_full_size_path, stride = 256)
"""