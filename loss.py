def ensembled_loss(y_true, y_pred):
    '''
    This loss is calculated as 
    class based IOU Loss + alpha*Instance based IOU Loss
    alpha=0.5
    '''
    alpha=1
    loss = iou_mean_loss(y_true, y_pred) + 1*ib_iou_loss(y_true, y_pred)
    return loss

def ib_iou_loss(y_true, y_pred):
    total_iou = 0
    for i in range(3):
        total_iou_for_one_class = 0
        total_instance_num_for_batch = 0
        y_true_tmp = y_true[:,:,:,i]
        y_pred_tmp = y_pred[:,:,:,i]
        #CONVERT EVERYTHING TO ARRAY
        true_arr = K.eval(y_true_tmp).astype('uint8')
        pred_arr = K.eval(y_pred_tmp).astype('uint8')
        intersection_arr = true_arr * pred_arr
        union_arr = true_arr + pred_arr - intersection_arr
        for j in range(3):  #BATCH SIZE
            union_arr_ = union_arr[j,:,:]
            y_true_tmp_ = y_true_tmp[j,:,:] 
            y_pred_tmp_ = y_pred_tmp[j,:,:]
            #Find every intance in the UNION SET 
            output = cv2.connectedComponentsWithStats(union_arr_, 1, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            for k in range(1, numLabels):
                # construct a mask for the current connected component and
                # then take the bitwise OR with the mask
                componentMask = (labels == k).astype("uint8") * 1
                #CONVERT MASK TO TENSOR
                mask_tensor = tf.convert_to_tensor(componentMask, dtype=tf.float32)
                mask_tensor_f = K.flatten(mask_tensor)
                y_true_tmp_f = K.flatten(y_true_tmp_)
                y_pred_tmp_f = K.flatten(y_pred_tmp_)
                #MASK BOTH INTERSECTION AND UNION
                intersection = K.sum(y_true_tmp_f * y_pred_tmp_f * mask_tensor_f, axis=-1)
                union = K.sum(y_true_tmp_f*mask_tensor_f, axis=-1)+K.sum(y_pred_tmp_f*mask_tensor_f, axis=-1)-intersection
                iou = K.mean((intersection) / (union ))
                total_iou_for_one_class = total_iou_for_one_class + iou
            total_instance_num_for_batch = total_instance_num_for_batch + numLabels -1
        if(total_instance_num_for_batch):
            total_iou = total_iou + (total_iou_for_one_class/total_instance_num_for_batch)
    #return tf.cast(3-total_iou, tf.float32)
    return tf.cast(-K.log(total_iou/3), tf.float32)

def iou_mean_loss(y_true, y_pred):
    smooth = 1e-7
    total_iou = 0
    for i in range(3):#CLASS NUM
        y_true_tmp = y_true[:,:,:,i]
        y_pred_tmp = y_pred[:,:,:,i]
        y_true_tmp_f = K.flatten(y_true_tmp)
        y_pred_tmp_f = K.flatten(y_pred_tmp)
        intersection = K.sum(y_true_tmp_f * y_pred_tmp_f, axis=-1)
        union = K.sum(y_true_tmp_f, axis=-1)+K.sum(y_pred_tmp_f, axis=-1)-intersection
        iou = K.mean((intersection + smooth) / (union + smooth))
        total_iou = total_iou + iou
        #print(iou, mean_iou)
    return -K.log(total_iou/3)

