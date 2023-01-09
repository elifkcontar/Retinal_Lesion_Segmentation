import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize
import pandas as pd
import time

from sklearn import metrics

from tensorflow.keras.utils import to_categorical   
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image

import augmentation as augment
from auprc import plot_pr_curve
import utils as util
from utils import  make_subfolder, show_images
from model import dice_loss, dice_coef_acc

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, jaccard_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import csv


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def visualize_loss_acc(history):
    history = {'loss': history['loss'], 'val_loss': history['val_loss'], 
        'accuracy': history['accuracy'], 'val_accuracy': history['val_accuracy']}


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    ax1.plot(history['loss'], label='Train loss')
    ax1.plot(history['val_loss'], label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(history['accuracy'], label='Train accuracy')
    ax2.plot(history['val_accuracy'], label='Validation accuracy')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    plt.show()

def create_undersampled_background(my_output, b_channel=3):
    '''
    Takes N channel 2D input array. Backgorund channel number is specified by b_chanel, 
    if not last channel is taken. Then additional undersampled background is added as new last channel
    '''
    additional_channel = np.zeros((my_output.shape[0],my_output.shape[1],1))
    ret_output = np.zeros((my_output.shape[0],my_output.shape[1],my_output.shape[2]+1))   #One more additional channel
    prev_output = my_output

    sample_num = int(my_output.shape[0]*my_output.shape[1]*0.01)   #0.001 is less sampled ratio
    sample_array = np.random.randint(my_output.shape[0], size=(sample_num, 2)) #2 for 2D array
    for i in range(sample_array.shape[0]):
        a=my_output[sample_array[i]]
        b=additional_channel[sample_array[i]]
        c=my_output[sample_array[i],3]
        additional_channel[sample_array[i]] = my_output[sample_array[i],3]
        prev_output[sample_array[i],3] = 0
    ret_output[:,:,-1] = prev_output
    ret_output[:,:,4] = additional_channel
    
    show_images([my_output[:,:,3], ret_output[:,:,3], ret_output[:,:,4], additional_channel])

def generator_array(x,y, batch_size=2, augmentation=False):
    '''
    Work with given input-output array
    '''
    c = 0
    while (True):
        img_batch = np.zeros((batch_size, x.shape[1],x.shape[2], x.shape[3]))
        mask_batch = np.zeros((batch_size, y.shape[1], y.shape[2], y.shape[3]))
        #sample_weight = np.zeros((batch_size, IMG_SIZE, IMG_SIZE))

        for i in range(c, c+batch_size):
            if(augmentation):
                img_batch[i-c], mask_batch[i-c] = augment.apply_augmentation(x[i,:,:,:], y[i,:,:,:])
            else:
                img_batch[i-c], mask_batch[i-c] = x[i,:,:,:], y[i,:,:,:]

        c+=batch_size
        if(c+batch_size>(x.shape[0])):
            c=0
            x, y = unison_shuffled_copies(x, y)

        yield img_batch, mask_batch

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def patched_predict(img, y, model, window_size = 512, stride = 256, channel = 5):
    bound = (window_size - stride)//2
    #print(img.shape[0],img.shape[1],img.shape[2])
    fullsize_pred = np.zeros((img.shape[0], img.shape[1], channel))
    #tmp_arr = np.full((img.shape[0], img.shape[1]), channel-1)
    #fullsize_pred = y_pred=to_categorical(tmp_arr, num_classes=channel)

    row=(img.shape[0] - 2*bound)//stride
    column=(img.shape[1] - 2*bound)//stride
    #model_class = util.load_model('/home/vivente/Desktop/TEZ/Code/segmentation/out_classification/2021_November_27-15_57_51/fcn_trained_model')
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
            y_patch = y[x_start:x_stop, y_start:y_stop, :-1]
            #part_pred_class = model_class.predict(np.reshape(pred_in, (1,window_size, window_size, 3)))
            #if(part_pred_class<0.5):
            part_pred = model.model.predict(np.reshape(pred_in, (1,window_size, window_size, 3)))
            #if(y_patch.max()!=0):
            #    part_pred = model.predict(np.reshape(pred_in, (1,window_size, window_size, 3)))
            #else:
            #    part_pred = background_array
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

def patched_predict_binary(img, y, model, window_size = 512, stride = 256, channel = 1):
    bound = (window_size - stride)//2
    fullsize_pred = np.zeros((img.shape[0], img.shape[1])).reshape(img.shape[0], img.shape[1],1)

    row=(img.shape[0] - 2*bound)//stride
    column=(img.shape[1] - 2*bound)//stride
    #model_class = util.load_model('/home/vivente/Desktop/TEZ/Code/segmentation/out_classification/2021_November_27-15_57_51/fcn_trained_model')
    #model_class = 0
    background_array = np.zeros((window_size, window_size)).reshape(window_size, window_size,1)

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

            pred_in = img[x_start:x_stop, y_start:y_stop, :]
            y_patch = y[x_start:x_stop, y_start:y_stop]
            #part_pred_class = model_class.predict(np.reshape(pred_in, (1,window_size, window_size, 3)))
            #if(part_pred_class<0.5):
            if(y_patch.max()!=0):
                #a = resize(pred_in,(256,256))
                part_pred = model.predict(np.reshape(pred_in, (1,window_size, window_size, 3)))
            else:
                part_pred = background_array
            part_pred = np.reshape(part_pred, (window_size, window_size, channel))
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
 

    return np.asarray(fullsize_pred)

def get_metric(model, test_df, x_test_path, y_test_path, config):
    IMG_SIZE = config.IMG_SIZE
    CLASS_NUM = config.CLASS_NUM
    acc_list=[]
    prec_list=[]
    recall_list=[]
    acc_list_=[]
    prec_list_=[]
    recall_list_=[]
    test_images = []
    test_labels = []
    test_labels_flat = []
    test_preds = []
    test_preds_flat = []
    sample_num = int(len(test_df['Input']))
    acc = np.zeros(CLASS_NUM)
    recall = np.zeros(CLASS_NUM)
    prec = np.zeros(CLASS_NUM)

    for i in range(sample_num):
        
        x = np.asarray(io.imread(x_test_path+test_df.loc[i,'Input']))
        x = np.reshape(x, (1,IMG_SIZE, IMG_SIZE, 3))/255.
        y_true = np.asarray(util.read_var(y_test_path+test_df.loc[i,'4CH']).astype(float))
        y = model.predict(x)
        y_pred = np.reshape(y, (IMG_SIZE*IMG_SIZE, CLASS_NUM))
        y_true = np.reshape(y_true, (IMG_SIZE*IMG_SIZE, CLASS_NUM))
        y_pred=np.argmax(y_pred, axis=-1)
        y_true=np.argmax(y_true, axis=-1)
        y_pred = to_categorical(y_pred, num_classes=CLASS_NUM)
        y_true = to_categorical(y_true, num_classes=CLASS_NUM)
        for j in range(CLASS_NUM):    
            acc[j] = acc[j] + accuracy_score(y_true[:,j], y_pred[:,j])
            recall[j] = recall[j] + recall_score(y_true[:,j], y_pred[:,j], average='binary') #same with sensitivity
            prec[j] = prec[j] + precision_score(y_true[:,j], y_pred[:,j], average='binary')

    acc_list = list(acc/sample_num)
    recall_list = list(recall/sample_num)
    prec_list = list(prec/sample_num)

    filename = "metrics.csv"
    fields = ['MA', 'HE', 'EX', 'BV', 'BG']
    with open(config.log_dir + filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerow(acc_list) 
        csvwriter.writerow(recall_list)
        csvwriter.writerow(prec_list)
    
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=[acc, recall, prec], colLabels=['MA', 'HE', 'EX', 'BV', 'BG'], rowLabels=['Acc', 'Prec', 'Recl'],loc='center')

    fig.tight_layout()
    plt.savefig(config.log_dir + "metrics.png")
    plt.show(block=False)

    return acc_list, recall_list, prec_list

def get_full_size_metric(model, test_df, x_test_path, y_test_path, config, name, stride = 256):
    IMG=io.imread(x_test_path+test_df.loc[0,'Input'])
    IMG_SIZE_0 = IMG.shape[0]
    IMG_SIZE_1 = IMG.shape[1]
    CLASS_NUM = config.CLASS_NUM
    prec_list=[]
    recall_list=[]
    f1_list =[]
    iou_list = []
    dice_list = []
    accuracy_list = []
    specificity_list = []
    aupr_list = []
    test_images = []
    test_labels = []
    test_preds = []
    sample_num = int(len(test_df['Input']))
  
    for i in range(sample_num):
        
        x=np.asarray(io.imread(x_test_path+test_df.loc[i,'Input']))
        x=x/255.
        y=np.asarray(util.read_var(y_test_path+test_df.loc[i,'4CH']).astype(float))
        y=np.asarray(y[:,:,:-1])
        pred = (patched_predict(x, y, model, window_size = config.IMG_SIZE, stride = stride, channel = CLASS_NUM))
        #pred = (patched_predict_binary(x, y, model, window_size = config.IMG_SIZE, stride = stride, channel = CLASS_NUM))

        y_pred = np.asarray(pred)
        y_true = np.asarray(y).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
        y_pred = np.asarray(y_pred).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
        
        #Convert one hot vector to integer label to get rid of gray values
        '''
        y_pred=np.argmax(y_pred, axis=-1)
        y_true=np.argmax(y_true, axis=-1)
        y_pred = to_categorical(y_pred, num_classes=CLASS_NUM)
        y_true = to_categorical(y_true, num_classes=CLASS_NUM)
        '''

        #prepare and save coloured full size images for visualitaion purposes
        pred_show = y_pred.copy()
        for k in range(CLASS_NUM):     
            pred_show[:,k] = y_pred[:,k] > 0.5
        pred_show = np.reshape(y_pred, (IMG_SIZE_0, IMG_SIZE_1, CLASS_NUM))
        true_show = np.reshape(y_true, (IMG_SIZE_0, IMG_SIZE_1, CLASS_NUM))
        colour_img_pred = util.colorize_binary_img(pred_show[:,:,:])
        colour_img_true = util.colorize_binary_img(true_show[:,:,:])
        util.save_images([colour_img_true,colour_img_pred, x, x],['True','Prediction','Original','Original'], 
                            save_dir = config.log_dir + 'visualisations/', name = name + "_full_size_" + str(i+55).zfill(2) + '.png')
        
        util.save_var(y_true, config.log_dir + 'prediction/'+'y_true_'+str(i+55))
        util.save_var(y_pred, config.log_dir + 'prediction/'+name+'_y_pred_'+str(i+55))
  
    for i in range(sample_num):
        test_labels.append(util.read_var(config.log_dir + 'prediction/'+'y_true_'+str(i+55)))
        test_preds.append(util.read_var(config.log_dir + 'prediction/'+name+'_y_pred_'+str(i+55)))
    
    #for j in range(CLASS_NUM-1):
    for j in range(CLASS_NUM):    
        test_labels = np.asarray(test_labels).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
        test_preds = np.asarray(test_preds).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)


        prec, recl, thresholds = precision_recall_curve(test_labels[:,j], test_preds[:,j])
        aupr = auc(recl, prec)
        aupr_list.append(aupr)
        #Get best threshold
        fscore = (2 * prec * recl) / (prec + recl)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        plot_pr_curve(prec, recl, 1)

        '''
        accuracy, specificity, dice, precision, recall, f1, iou = my_metrics(test_labels[:,j], test_preds[:,j])
        accuracy_list.append(accuracy)
        specificity_list.append(specificity)
        dice_list.append(dice)
        recall_list.append(recall)
        prec_list.append(precision)
        f1_list.append(f1)
        iou_list.append(iou)
        '''

        time.sleep(0.2)

        
    filename = name + "_full_size_metrics.csv"
    fields = ['MA', 'HE', 'EX']
    with open(config.log_dir + filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)  
        #csvwriter.writerow(accuracy_list)  
        #csvwriter.writerow(specificity_list)  
        #csvwriter.writerow(dice_list) 
        #csvwriter.writerow(recall_list) 
        #csvwriter.writerow(prec_list) 
        #csvwriter.writerow(f1_list) 
        #csvwriter.writerow(iou_list)
        csvwriter.writerow(aupr_list) 

    '''
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=[accuracy_list, specificity_list, dice_list, recall_list, prec_list, f1_list, iou_list], colLabels=['MA', 'HE', 'EX'], rowLabels=['Acc', 'Spec', 'Dice', 'Recl', 'Prec', 'f-1', 'IoU'],loc='center')

    fig.tight_layout()
    plt.savefig(config.log_dir + name + "_full_size_metrics.png")
    #plt.show(block=False)
    plt.close(fig)
    '''

    return 0

def get_full_size_metric_binary(model, test_df, x_test_path, y_test_path, config, name, stride = 256):
    IMG=io.imread(x_test_path+test_df.loc[0,'Input'])
    IMG_SIZE_0 = IMG.shape[0]
    IMG_SIZE_1 = IMG.shape[1]
    CLASS_NUM = config.CLASS_NUM
    prec_list=[]
    recall_list=[]
    f1_list =[]
    iou_list = []
    dice_list = []
    accuracy_list = []
    specificity_list = []
    aucpr_list = []
    test_images = []
    test_labels = []
    test_preds = []
    sample_num = int(len(test_df['Input']))
    '''
    for i in range(sample_num):
        
        x=np.asarray(io.imread(x_test_path+test_df.loc[i,'Input']))
        x=x/255.
        y=np.asarray(util.read_var(y_test_path+test_df.loc[i,'4CH']).astype(float))
        y = y[:,:,0]

        #pred = (patched_predict(x, y, model, window_size = config.IMG_SIZE, stride = stride, channel = CLASS_NUM))
        pred = (patched_predict_binary(x, y, model, window_size = config.IMG_SIZE, stride = stride, channel = CLASS_NUM))

        y_pred = np.asarray(pred)
        y_true = np.asarray(y).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
        y_pred = np.asarray(y_pred).reshape(IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)

        #prepare and save coloured full size images for visualitaion purposes
        pred_show = np.reshape(y_pred, (IMG_SIZE_0, IMG_SIZE_1, CLASS_NUM))
        true_show = np.reshape(y_true, (IMG_SIZE_0, IMG_SIZE_1, CLASS_NUM))
        colour_img_pred = util.colorize_binary_img(pred_show[:,:,:])
        colour_img_true = util.colorize_binary_img(true_show[:,:,:])
        util.save_images([colour_img_true,colour_img_pred, x, x],['True','Prediction','Original','Original'], 
                            save_dir = config.log_dir + 'visualisations/', name = name + "_full_size_" + str(i+55).zfill(2) + '.png')
        
        util.save_var(y_true, config.log_dir + 'prediction/'+'y_true_'+str(i+55))
        util.save_var(y_pred, config.log_dir + 'prediction/'+name+'_y_pred_'+str(i+55))
    '''
    for i in range(sample_num):
        #test_labels.append(util.read_var(config.log_dir + 'prediction/'+'y_true_'+str(i+55)))
        #test_preds.append(util.read_var(config.log_dir + 'prediction/'+name+'_y_pred_'+str(i+55)))

        test_labels.append(util.read_var('/home/vivente/Desktop/TEZ/Code/segmentation/out/2022_January_15-18_26_53/prediction/'+'y_true_'+str(i+55)))
        test_preds.append(util.read_var('/home/vivente/Desktop/TEZ/Code/segmentation/out/2022_January_15-18_26_53/prediction/'+name+'_y_pred_'+str(i+55)))
    
    for j in range(CLASS_NUM):    
        test_labels = np.asarray(test_labels).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
        test_preds = np.asarray(test_preds).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)

        prec, recl, thresholds = precision_recall_curve(test_labels, test_preds)
        my_auc = auc(recl, prec)

        fscore = (2 * prec * recl) / (prec + recl)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

        accuracy, specificity, dice, precision, recall, f1, iou = my_metrics_binary(test_labels, test_preds, thresholds[ix])
        accuracy_list.append(accuracy)
        specificity_list.append(specificity)
        dice_list.append(dice)
        recall_list.append(recall)
        prec_list.append(precision)
        f1_list.append(f1)
        iou_list.append(iou)
        aucpr_list.append(my_auc)
        print(my_auc)
        time.sleep(0.2)

        
    filename = name + "_full_size_metrics.csv"
    fields = ['MA', 'HE', 'EX']
    with open(config.log_dir + filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)  
        csvwriter.writerow(accuracy_list)  
        csvwriter.writerow(specificity_list)  
        csvwriter.writerow(dice_list) 
        csvwriter.writerow(recall_list) 
        csvwriter.writerow(prec_list) 
        csvwriter.writerow(f1_list) 
        csvwriter.writerow(iou_list) 
        csvwriter.writerow(aucpr_list) 
    
    return 0

def my_metrics_binary(y_true, y_pred, thresh):
    y_pred = y_pred > thresh
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    #####from confusion matrix calculate accuracy
    accuracy=(tp+tn)/(tn+ fp+ fn+ tp)

    specificity = tn/(tn+fp)

    dice_score = (2*tp) / (2*tp + fp + fn)

    precision = tp/(tp+fp)
    
    recall = tp/(tp+fn)
    
    f1 = 2*precision*recall/(precision+recall)

    iou = tp/(tp+fp+fn)

    return accuracy, specificity, dice_score, precision, recall, f1, iou

def my_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    #####from confusion matrix calculate accuracy
    accuracy=(tp+tn)/(tn+ fp+ fn+ tp)

    specificity = tn/(tn+fp)

    dice_score = (2*tp) / (2*tp + fp + fn)

    precision = tp/(tp+fp)
    
    recall = tp/(tp+fn)
    
    f1 = 2*precision*recall/(precision+recall)

    iou = tp/(tp+fp+fn)

    return accuracy, specificity, dice_score, precision, recall, f1, iou

def dice_score(y_true, y_pred):
    num = np.sum(y_true * y_pred)
    denum = np.sum(y_true) + np.sum(y_pred)
    return (2*num)/denum
'''
def get_full_size_metric(model, test_df, x_test_path, y_test_path, config, name, stride = 256):
    IMG=io.imread(x_test_path+test_df.loc[0,'Input'])
    IMG_SIZE_0 = IMG.shape[0]
    IMG_SIZE_1 = IMG.shape[1]
    CLASS_NUM = config.CLASS_NUM
    acc_list=[]
    prec_list=[]
    recall_list=[]
    iou_list = []
    test_images = []
    test_labels = []
    test_preds = []
    sample_num = int(len(test_df['Input']))

    for i in range(sample_num):
        
        x=np.asarray(io.imread(x_test_path+test_df.loc[i,'Input']))
        x=x/255.
        y=np.asarray(util.read_var(y_test_path+test_df.loc[i,'4CH']).astype(float))
        test_images.append(x)
        test_labels.append(y)
        test_preds.append(patched_predict(x, model, window_size = config.IMG_SIZE, stride = stride, channel = CLASS_NUM))

    y_pred = np.asarray(test_preds)
    y_true = np.asarray(test_labels).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
    y_pred = np.asarray(y_pred).reshape(sample_num*IMG_SIZE_0*IMG_SIZE_1,CLASS_NUM)
    
    #Convert one hot vector to integer label to get rid of gray values
    y_pred=np.argmax(y_pred, axis=-1)
    y_true=np.argmax(y_true, axis=-1)
    y_pred = to_categorical(y_pred, num_classes=CLASS_NUM)
    y_true = to_categorical(y_true, num_classes=CLASS_NUM)

    #prepare and save coloured full size images for visualitaion purposes
    pred_show = np.reshape(y_pred, (sample_num, IMG_SIZE_0, IMG_SIZE_1, CLASS_NUM))
    true_show = np.reshape(y_true, (sample_num, IMG_SIZE_0, IMG_SIZE_1, CLASS_NUM))
    for ii in range(sample_num):
        colour_img_pred = util.colorize_binary_img(pred_show[ii,:,:,:])
        colour_img_true = util.colorize_binary_img(true_show[ii,:,:,:])
        util.save_images([colour_img_true,colour_img_pred, test_images[ii], test_images[ii]],['True','Prediction','Original','Original'], 
                        save_dir = config.log_dir + 'visualisations/', name = name + "_full_size_" + str(ii).zfill(2) + '.png')
    
    for j in range(CLASS_NUM):    
        acc=accuracy_score(y_true[:,j], y_pred[:,j])
        recall=recall_score(y_true[:,j], y_pred[:,j], average='binary') #same with sensitivity
        prec=precision_score(y_true[:,j], y_pred[:,j], average='binary')
        iou=jaccard_score(y_true[:,j], y_pred[:,j], average='binary')
        acc_list.append(acc)
        recall_list.append(recall)
        prec_list.append(prec)
        iou_list.append(iou)

        
    filename = name + "_full_size_metrics.csv"
    fields = ['MA', 'HE', 'EX', 'BG']
    with open(config.log_dir + filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerow(acc_list) 
        csvwriter.writerow(recall_list) 
        csvwriter.writerow(prec_list) 
        csvwriter.writerow(iou_list) 

    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=[acc_list, recall_list, prec_list, iou_list], colLabels=['MA', 'HE', 'EX', 'BG'], rowLabels=['Acc', 'Recl', 'Prec', 'IoU'],loc='center')

    fig.tight_layout()
    plt.savefig(config.log_dir + name + "_full_size_metrics.png")
    #plt.show(block=False)
    plt.close(fig)

    return acc_list, recall_list, prec_list, iou_list
'''

def save_metrics(config, loss, accuracy, acc, recall, prec, fs_acc, fs_recall, fs_prec):
    #header = ["log_dir", "WEIGHT_LIST", "Loss", "Accuracy", "Metric_Type", "MA", "HE", "EX", "BV", "BG"]
    rows = []
    row_1 = [config.log_dir, config.WEIGHT_LIST, loss, accuracy, "Acc"]
    row_2 = [config.log_dir, config.WEIGHT_LIST, loss, accuracy, "Recall"]
    row_3 = [config.log_dir, config.WEIGHT_LIST, loss, accuracy, "Prec"]
    row_4 = [config.log_dir, config.WEIGHT_LIST, loss, accuracy, "Acc"]
    row_5 = [config.log_dir, config.WEIGHT_LIST, loss, accuracy, "Recall"]
    row_6 = [config.log_dir, config.WEIGHT_LIST, loss, accuracy, "Prec"]
    rows.append(row_1.extend(acc))
    rows.append(row_2.extend(recall))
    rows.append(row_3.extend(prec))
    rows.append(row_4.extend(fs_acc))
    rows.append(row_5.extend(fs_recall))
    rows.append(row_6.extend(fs_prec))
    with open(r'out/Train_Metrics.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows([row_1,row_2,row_3,row_4,row_5,row_6])
    
def visualize_prediction(model_name, test_df, x_test_path, y_test_path, config, mode='both', save = False):
    '''
    Mode is gray or binary colored.
    '''
    IMG_SIZE = config.IMG_SIZE
    CLASS_NUM = config.CLASS_NUM
    save_dir = make_subfolder("visualisations/", config.log_dir)
    stop_condition = 256
    #model=load_model(model_name)
    model=model_name
    for i in range(int(len(test_df['Input']))):
    #for i in range(50):
        
        name = test_df.loc[i,'Input']
        x=io.imread(x_test_path+test_df.loc[i,'Input'])
        x=x/255.
        #y_true=np.asarray(util.read_var(y_test_path+test_df.loc[i,'4CH']).astype(float))
        y_true=np.asarray(util.read_var(y_test_path+test_df.loc[i,'4CH']).astype(float))
        y_pred = model.predict(np.reshape(x, (1,IMG_SIZE, IMG_SIZE, 3)))

        y_true=y_true.reshape(1*IMG_SIZE*IMG_SIZE, CLASS_NUM)
        y_pred=np.asarray( y_pred).reshape(1*IMG_SIZE*IMG_SIZE, CLASS_NUM)
        
        y_true_show=np.reshape(y_true, (IMG_SIZE, IMG_SIZE, CLASS_NUM))
        y_pred_show=np.reshape(y_pred, (IMG_SIZE, IMG_SIZE, CLASS_NUM))
        x_show=np.reshape(x, (IMG_SIZE, IMG_SIZE, 3))

        #Convert one hot vector to integer label to get rid of gray values
        y_pred=np.argmax(y_pred, axis=-1)
        y_true=np.argmax(y_true, axis=-1)
        y_pred = to_categorical(y_pred, num_classes=CLASS_NUM)
        y_true = to_categorical(y_true, num_classes=CLASS_NUM)
        pred_show = np.reshape(y_pred, (IMG_SIZE, IMG_SIZE, CLASS_NUM))
        true_show = np.reshape(y_true, (IMG_SIZE, IMG_SIZE, CLASS_NUM))
        if(save):
            if(mode=='gray'):
                util.save_images([y_pred_show[:,:,0], y_pred_show[:,:,1], y_pred_show[:,:,2], x_show, y_true_show[:,:,0], y_true_show[:,:,1], y_true_show[:,:,2], x_show], save_dir = save_dir, name = name)
            elif(mode=='binary'):
                util.save_images([util.colorize_binary_img(pred_show), x_show, util.colorize_binary_img(true_show), x_show], save_dir = save_dir, name = name)
            elif(mode=='both'):
                util.save_images([y_pred_show[:,:,0], y_pred_show[:,:,1], y_pred_show[:,:,2], util.colorize_binary_img(pred_show), 
                                y_true_show[:,:,0], y_true_show[:,:,1], y_true_show[:,:,2], util.colorize_binary_img(true_show)], titles='PPPPTTTT', save_dir = save_dir, name = name) 

        else:
            if(mode=='gray'):
                util.show_images([y_pred_show[:,:,0], y_pred_show[:,:,1], y_pred_show[:,:,2], x_show, y_true_show[:,:,0], y_true_show[:,:,1], y_true_show[:,:,2], x_show])
            elif(mode=='binary'):
                util.show_images([util.colorize_binary_img(pred_show), x_show, util.colorize_binary_img(true_show), x_show])
            elif(mode=='both'):
                util.show_images([y_pred_show[:,:,0], y_pred_show[:,:,1], y_pred_show[:,:,2], util.colorize_binary_img(pred_show), 
                                y_true_show[:,:,0], y_true_show[:,:,1], y_true_show[:,:,2], util.colorize_binary_img(true_show)], titles='PPPPTTTT') 
        if(i>stop_condition):
            break

def generator(dataframe, x_path, y_path, config, augmentation=True, mode='multi'):
    '''
    At the first stpe use mode=binary'. At the second step use mdoe='multi'
    '''
    IMG_SIZE = config.IMG_SIZE
    CLASS_NUM = config.CLASS_NUM
    batch_size = config.BATCH_SIZE
    c = 0
    dataframe = dataframe.sample(frac=1)  #Shuffle dataset using sample method
    while (True):
        x_batch = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3))
        y_batch = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, CLASS_NUM))
        #sample_weight = np.zeros((batch_size, IMG_SIZE, IMG_SIZE))

        for i in range(c, c+batch_size):
            x = io.imread(x_path+dataframe.loc[i,'Input'])
            #x = resize(x, (256,256))
            x = image.img_to_array(x)
            y_tmp = np.asarray(util.read_var(y_path+dataframe.loc[i,'4CH']).astype(float))
            if(mode=='binary'):
                #y = np.asarray(util.rgb2bool(util.read_image(y_path+dataframe.loc[i,'MA'])).astype(float))                
                y = np.reshape(y_tmp[:,:,0], (IMG_SIZE, IMG_SIZE, CLASS_NUM))
            elif(mode=='multi'):
                y = y_tmp
            elif(mode=='multi_pos'):
                y = np.reshape(y_tmp[:,:,:-1], (IMG_SIZE, IMG_SIZE, CLASS_NUM))
            if(augmentation):
                x_, y_  = augment.apply_augmentation(x, y)
                x_batch[i-c] = image.img_to_array(x_)/255.
                y_batch[i-c] = y_.astype(float)
            else:
                x_batch[i-c], y_batch[i-c] = image.img_to_array(x)/255., y

        c+=batch_size
        if(c+batch_size>len(dataframe.index)):
            c=0
            dataframe = dataframe.sample(frac=1)
        '''
        if(class_weight):                           #150 These values work for 'categorical_crossentopy_loss' function.
            y_batch[:,:,:,0]=y_batch[:,:,:,0]*weight_list[0]     #MA
            y_batch[:,:,:,1]=y_batch[:,:,:,1]*weight_list[1]     #HE
            y_batch[:,:,:,2]=y_batch[:,:,:,2]*weight_list[2]     #EX
            y_batch[:,:,:,3]=y_batch[:,:,:,3]*weight_list[3]     #BG
            #y_batch[:,:,:,4]=y_batch[:,:,:,4]*weight_list[4]     #BG
        '''

        yield x_batch, y_batch

def plot_history(history):
    '''
    Takes history dictionary and plot accuracy and loss graph
    '''
    plot_acc(history)
    plot_loss(history)

def plot_loss_and_acc(history, save=False, saveDir='out/'):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    ax1.plot(history['loss'], label='Train loss')
    ax1.plot(history['val_loss'], label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(history['accuracy'], label='Train accuracy')
    ax2.plot(history['val_accuracy'], label='Validation accuracy')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    if(save):
        plt.savefig(saveDir + 'loss_and_acc.png')
    #plt.show(block=False)
    plt.close(fig)

def plot_loss_and_acc_train(history, save=False, saveDir='out/'):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    ax1.plot(history['loss'], label='Train loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    #ax2.plot(history['acc'], label='Train accuracy')
    #ax2.legend(loc='best')
    #ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    if(save):
        plt.savefig(saveDir + 'loss_and_acc.png')
    plt.close(fig)

def plot_acc(history):
    '''
    TODO:Instead of history whats should be the input?
    Takes array of Plot training & validation accuracy values
    '''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(history):
    '''
    TODO:Instead of history whats should be the input?
    Plot training & validation loss values
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def adem_sil_bunu(my_dir):
    my_model = util.load_model(my_dir + "trained_model")
    my_config = util.read_var(my_dir + "config")

    patched_dataset_path = '/home/vivente/Desktop/DR/Data/Patched_dataset_512x512_stride_256/'
    test_df_path = patched_dataset_path + 'test.csv'
    x_test_path = patched_dataset_path + '1. Original Images/d. Bens Testing Set/'
    y_test_path = patched_dataset_path + '2. All Segmentation Groundtruths/b. Testing Set/0. All/'
    test_df=pd.read_csv(test_df_path)

    full_size_df_path = '/home/vivente/Desktop/DR/Data/dataset_2048x2048_bens_4ch/test.csv'
    x_full_size_path = '/home/vivente/Desktop/DR/Data/dataset_2048x2048_bens_4ch/1. Original Images/d. Bens Testing Set/'
    y_full_size_path = '/home/vivente/Desktop/DR/Data/dataset_2048x2048_bens_4ch/2. All Segmentation Groundtruths/b. Testing Set/0. All/'
    full_size_df = pd.read_csv(full_size_df_path)

    print("Metric calculation")
    acc, recall, prec = get_metric(my_model, test_df, x_test_path, y_test_path, my_config)
    print("Full size metric calculation")
    fs_acc, fs_recall, fs_prec = get_full_size_metric(my_model, full_size_df, x_full_size_path, y_full_size_path, my_config, stride = 256)
    save_metrics(my_config, 1, 0.95, acc, recall, prec, fs_acc, fs_recall, fs_prec)

if __name__ == "__main__":
    #my_dir = 'out/2020_October_23-20_09_35/'
    #adem_sil_bunu(my_dir)
    my_dir = 'out/2020_October_24-17_33_39/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_Ekim_25-10_11_16/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_Ekim_26-02_51_15/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_Ekim_26-19_32_08/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_October_28-18_23_20/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_Ekim_29-11_00_10/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_Ekim_30-03_43_16/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_Kasım_04-12_12_44/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_November_06-15_27_58/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_November_07-22_58_11/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_November_10-09_33_13/'
    adem_sil_bunu(my_dir)
    my_dir = 'out/2020_Kasım_11-01_39_29/'
    adem_sil_bunu(my_dir)
    #util.send_as_mail(my_config.log_dir)
