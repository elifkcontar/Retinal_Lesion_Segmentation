from sklearn import metrics
import tensorflow as tf
#import tensorflow_addons as tfa
#import tensorflow_addons.optimizers 
#from tensorflow_addons.optimizers  import GradientAccumulator as accum_opt
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, SeparableConv2D, AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
import numpy as np
import cv2
#from accumulated_opt import AccumOptimizer
#from acc_grad import convert_to_accumulate_gradient_optimizer
#from tensorflow.keras.applications.resnet.ResNet101 import ResNet101

def basic_unet(input_size=(512, 512, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=SGD(lr=5e-3), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def residual_unet(input_size=(512, 512, 1), out_size=3, start_neurons=16, DropoutRatio=0.5):
    inputs = Input(input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)


    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    output_layer_noActi = Conv2D(out_size, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('softmax')(output_layer_noActi)

    model = Model(inputs, output_layer)

    model.compile(optimizer=Adam(lr=1e-4), loss=ensembled_loss, metrics=['accuracy'],
                  run_eagerly=True)

    return model

def conv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):
    x = Conv2D(filters=num_filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               strides=strides,
                               dilation_rate=dilation_rate,
                               kernel_initializer=w_init,
                               use_bias=False)(tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def sepconv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):
    x = SeparableConv2D(filters=num_filters,
                                        depth_multiplier=1,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        strides=strides,
                                        dilation_rate=dilation_rate,
                                        depthwise_initializer=w_init,
                                        use_bias=False)(tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def JPU(endpoints: list, out_channels=512):
    h, w = endpoints[1].shape.as_list()[1:3]
    for i in range(1, 4):
        endpoints[i] = conv_block(endpoints[i], out_channels, 3)
        if i != 1:
            h_t, w_t = endpoints[i].shape.as_list()[1:3]
            scale = (h // h_t, w // w_t)
            endpoints[i] = UpSampling2D(
                size=scale, interpolation='bilinear')(endpoints[i])
    yc = Concatenate(axis=-1)(endpoints[1:])
    ym = []
    for rate in [1, 2, 4, 8]:
        ym.append(sepconv_block(yc, 512, 3, dilation_rate=rate))
    y = Concatenate(axis=-1)(ym)
    return endpoints, y

def ASPP(tensor):
    dims = tensor.shape.as_list()

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = conv_block(y_pool, num_filters=256, kernel_size=1)

    h_t, w_t = y_pool.shape.as_list()[1:3]
    scale = dims[1] // h_t, dims[2] // w_t
    y_pool = UpSampling2D(
        size=scale, interpolation='bilinear')(y_pool)

    y_1 = conv_block(tensor, num_filters=256, kernel_size=1, dilation_rate=1)
    y_6 = conv_block(tensor, num_filters=256, kernel_size=3, dilation_rate=6)
    y_6.set_shape([None, dims[1], dims[2], 256])
    y_12 = conv_block(tensor, num_filters=256, kernel_size=3, dilation_rate=12)
    y_12.set_shape([None, dims[1], dims[2], 256])
    y_18 = conv_block(tensor, num_filters=256, kernel_size=3, dilation_rate=18)
    y_18.set_shape([None, dims[1], dims[2], 256])

    y = Concatenate(axis=-1)([y_pool, y_1, y_6, y_12, y_18])
    y = conv_block(y, num_filters=256, kernel_size=1)
    return y

def fast_fcn(img_height=1024, img_width=1024, nclasses=19):
    base_model = ResNet101(include_top=False,
                           input_shape=[img_height, img_width, 3],
                           weights='imagenet')
    endpoint_names = ['conv2_block3_out', 'conv3_block4_out',
                      'conv4_block23_out', 'conv5_block3_out']
    endpoints = [base_model.get_layer(x).output for x in endpoint_names]

    _, image_features = JPU(endpoints)

    x_a = ASPP(image_features)
    h_t, w_t = x_a.shape.as_list()[1:3]
    #scale = (img_height / 4) // h_t, (img_width / 4) // w_t
    scale = 2, 2
    print(scale)
    x_a = UpSampling2D(
        size=scale, interpolation='bilinear')(x_a)

    x_b = base_model.get_layer('conv2_block3_out').output
    x_b = conv_block(x_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([x_a, x_b])
    x = conv_block(x, num_filters=256, kernel_size=3)
    x = conv_block(x, num_filters=256, kernel_size=3)
    h_t, w_t = x.shape.as_list()[1:3]
    scale = img_height // h_t, img_width // w_t
    x = UpSampling2D(size=scale, interpolation='bilinear')(x)

    x = Conv2D(nclasses, (1, 1), name='output_layer', activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x, name='JPU')
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    model.compile(loss=ensembled_loss,
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'],
                  run_eagerly=True)
    return model

class FastFCN():
    def __init__(self,img_height=1024, img_width=1024, nclasses=19):
        self.base_model = tf.keras.applications.resnet.ResNet101(include_top=False,
                            input_shape=[img_height, img_width, 3],
                            weights='imagenet')
        endpoint_names = ['conv2_block3_out', 'conv3_block4_out',
                        'conv4_block23_out', 'conv5_block3_out']
        endpoints = [self.base_model.get_layer(x).output for x in endpoint_names]

        _, image_features = JPU(endpoints)

        x_a = ASPP(image_features)
        h_t, w_t = x_a.shape.as_list()[1:3]
        #scale = (img_height / 4) // h_t, (img_width / 4) // w_t
        scale = 2, 2
        print(scale)
        x_a = UpSampling2D(
            size=scale, interpolation='bilinear')(x_a)

        x_b = self.base_model.get_layer('conv2_block3_out').output
        x_b = conv_block(x_b, num_filters=48, kernel_size=1)

        x = Concatenate(axis=-1)([x_a, x_b])
        x = conv_block(x, num_filters=256, kernel_size=3)
        x = conv_block(x, num_filters=256, kernel_size=3)
        h_t, w_t = x.shape.as_list()[1:3]
        scale = img_height // h_t, img_width // w_t
        x_elif = UpSampling2D(size=scale, interpolation='bilinear')(x)

        x = Conv2D(nclasses+1, (1, 1), name='output_layer', activation='sigmoid')(x_elif)
        self.model_old = Model(inputs=self.base_model.input, outputs=x, name='JPU_old')
        for layer in self.model_old.layers:
            if isinstance(layer, BatchNormalization):
                layer.momentum = 0.9997
                layer.epsilon = 1e-5
            elif isinstance(layer, Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)   

        #self.model_old.load_weights("fcn_trained_model.h5")

        x_2 =Conv2D(nclasses, (1, 1), name='output_layer', activation='sigmoid')(x_elif)
        self.model = Model(inputs=self.base_model.input, outputs=x_2, name='JPU') 
    
    def set_trainable(self, type='True'):
        self.model_old.trainable = type
    
    def compile(self):
        #opt = accum_opt(Adam(learning_rate=1e-3), accum_steps=2)
        opt = Adam(learning_rate=1e-4)
        self.model.compile(loss=ensembled_loss,
                    optimizer = opt,
                    #metrics = [AUC(curve='PR')],
                    run_eagerly=True)

def my_acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(
    y_true[:,:,:,:-1], y_pred[:,:,:,:-1])

def weighted_categorical_crossentropy(y_true, y_pred):
    # y_true is a matrix of weight-hot vectors (like 1-hot, but they have weights instead of 1s)
    y_true_mask = K.one_hot(K.argmax(y_true, axis=-1), 4)  # [0 0 W 0] -> [0 0 1 0] where W >= 1.
    cce = K.categorical_crossentropy(y_pred, y_true_mask)  # one dim less (each 1hot vector -> float number)
    y_true_weights_maxed = K.max(y_true, axis=-1)  # [0 120 0 0] -> 120 - get weight for each weight-hot vector
    wcce = cce*y_true_weights_maxed
    return K.mean(wcce)       

def ensembled_loss(y_true, y_pred):
    '''
    This loss is calculated as 
    class based IOU Loss + alpha*Instance based IOU Loss
    alpha=0.5
    '''
    alpha=0.5
    loss = cce = iou_mean_loss(y_true, y_pred) + 1*ib_iou_loss(y_true, y_pred)
    return loss

def ensembled_loss_binary(y_true, y_pred):
    '''
    This loss is calculated as 
    class based IOU Loss + alpha*Instance based IOU Loss
    alpha=0.5
    '''
    alpha=0.5
    loss = cce = iou_mean_loss_binary(y_true, y_pred) + 1*ib_iou_loss_binary(y_true, y_pred)
    return loss

def ib_iou_loss(y_true, y_pred):
    total_iou = 0
    smooth = 1e-7
    for i in range(3):#CLASS NUMBER
        total_iou_for_one_class = 0
        total_instance_num_for_batch = 0
        y_true_tmp = y_true[:,:,:,i]
        y_pred_tmp = y_pred[:,:,:,i]
        #CONVERT EVERYTHING TO ARRAY
        true_arr = K.eval(y_true_tmp).astype('uint8')
        pred_arr = K.eval(y_pred_tmp).astype('uint8')

        intersection_arr = true_arr * pred_arr
        union_arr = true_arr + pred_arr - intersection_arr
        for j in range(2):  #BATCH SIZE
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
    return tf.cast(-K.log((total_iou+smooth)/3), tf.float32)

def ib_iou_loss_binary(y_true, y_pred):
    total_iou = 0
    smooth = 1e-7

    total_iou_for_one_class = 0
    total_instance_num_for_batch = 0
    
    true_arr = K.eval(y_true).astype('uint8')
    pred_arr = K.eval(y_pred).astype('uint8')

    intersection_arr = true_arr * pred_arr
    union_arr = true_arr + pred_arr - intersection_arr
    for j in range(2):  #BATCH SIZE
        union_arr_ = union_arr[j,:,:]

        y_true_tmp_ = y_true[j,:,:] 
        y_pred_tmp_ = y_pred[j,:,:]

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

    return tf.cast(-K.log((total_iou+smooth)/3), tf.float32)

'''
def iou_loss(y_true, y_pred):
    y_true_tmp = y_true[:-1]
    y_pred_tmp = y_pred[:-1]
    smooth = 1
    intersection = K.sum(K.abs(y_true_tmp * y_pred_tmp), axis=-1)
    union = K.sum(y_true_tmp,-1)+K.sum(y_pred_tmp,-1)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1-iou
'''

def iou_loss(y_true, y_pred):
    y_true_tmp = y_true[:,:,:,:-1]
    y_pred_tmp = y_pred[:,:,:,:-1]
    y_true_tmp_f = K.flatten(y_true_tmp)
    y_pred_tmp_f = K.flatten(y_pred_tmp)
    smooth = 1e-7
    intersection = K.sum(y_true_tmp_f * y_pred_tmp_f, axis=-1)
    print(intersection.shape)
    union = K.sum(y_true_tmp_f, axis=-1)+K.sum(y_pred_tmp_f, axis=-1)-intersection
    print(union.shape)
    iou = K.mean((intersection + smooth) / (union + smooth))
    print(iou.shape)

    return 1-iou

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

def iou_mean_loss_binary(y_true, y_pred):
    smooth = 1e-7
    total_iou = 0
    
    y_true_tmp_f = K.flatten(y_true)
    y_pred_tmp_f = K.flatten(y_pred)

    intersection = K.sum(y_true_tmp_f * y_pred_tmp_f, axis=-1)
    union = K.sum(y_true_tmp_f, axis=-1)+K.sum(y_pred_tmp_f, axis=-1)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    total_iou = total_iou + iou
    return -K.log(total_iou/3)

def dice_loss(y_true, y_pred):
    y_true_tmp = y_true[:-1]
    y_pred_tmp = y_pred[:-1]
    smooth=1
    intersection = K.sum(K.abs(y_true_tmp * y_pred_tmp), axis=-1)
    dice = (2. * intersection + smooth) / (K.sum(K.square(y_true_tmp),-1) + K.sum(K.square(y_pred_tmp),-1) + smooth)
    return 1-dice

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)    #improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def dice_coef_acc():
    def accuracy(y_true, y_pred):
        smooth=1
        y_true_mask = K.one_hot(K.argmax(y_true, axis=-1), 5)  # [0 0 W 0] -> [0 0 1 0] where W >= 1.
        intersection = K.sum(y_true_mask * y_pred, axis=[1,2,3])
        union = K.sum(y_true_mask, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice
    return accuracy