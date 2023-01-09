import skimage as sk
import skimage.io as io
import numpy as np
import random
import utils as util
import os
from skimage.transform import resize
from skimage import img_as_ubyte
import csv
from math import ceil


#Augmentation functions
def anticlockwise_rotation(image, label):
    angle= random.randint(0,180)
    img_out = sk.transform.rotate(image, angle, mode = 'symmetric')
    if(label.ndim == 3): #not just a simple image but it is multichannel image
        lbl_out = np.zeros([label.shape[0],label.shape[1],label.shape[2]], dtype = bool)
        for ch in range (label.shape[2]):
            lbl_out[:,:,ch] = sk.transform.rotate(label[:,:,ch], angle, mode = 'symmetric') > 0.5
    else: # simple mask
        lbl_out = sk.transform.rotate(label, angle, mode = 'symmetric') > 0.5
    return img_out, lbl_out

def clockwise_rotation(image, label):
    angle= random.randint(0,180)
    img_out = sk.transform.rotate(image, -angle, mode = 'symmetric')
    if(label.ndim == 3): #not just a simple image but it is multichannel image
        lbl_out = np.zeros([label.shape[0],label.shape[1],label.shape[2]], dtype = bool)
        for ch in range (label.shape[2]):
            lbl_out[:,:,ch] = sk.transform.rotate(label[:,:,ch], -angle, mode = 'symmetric') > 0.5
    else: # simple mask
        lbl_out = sk.transform.rotate(label, -angle, mode = 'symmetric') > 0.5
    return img_out, lbl_out

def h_flip(image, label):
    return  np.fliplr(image), np.fliplr(label)

def v_flip(image, label):
    return np.flipud(image), np.flipud(label)

def add_noise(image, label):
   return sk.util.random_noise(image), label

def shift(image, label):
    limit = image.shape[0]//8
    dx= random.randint(-limit,limit)
    dy= random.randint(-limit,limit)
    transform = sk.transform.AffineTransform(translation = (dx, dy))
    warp_image = sk.transform.warp(image, transform, mode = 'symmetric')
    if(label.ndim == 3): #not just a simple image but it is multichannel image
        lbl_out = np.zeros([label.shape[0],label.shape[1],label.shape[2]], dtype = bool)
        for ch in range (label.shape[2]):
            lbl_out[:,:,ch] = sk.transform.warp(label[:,:,ch], transform, mode = 'symmetric')> 0.5
    else: # simple mask
        lbl_out = sk.transform.warp(label, transform, mode = 'symmetric')> 0.5
    return warp_image, lbl_out

def change_brightness(image, label):   
    gamma = random.random() + 0.5
    image_bright = sk.exposure.adjust_gamma(image, gamma ,gain=1)
    return image_bright, label

def apply_augmentation(img, label):
    transformations = {'rotate anticlockwise': anticlockwise_rotation,
                      'rotate clockwise': clockwise_rotation,
                      'horizontal flip': h_flip, 
                      'vertical flip': v_flip
                      #'adding noise': add_noise,
                      #'shift image': shift,
                      #'brightness': change_brightness
                 }                #use dictionary to store names of functions 
    n = 0       #variable to iterate till number of transformation to apply
    transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image

    while n <= transformation_count:
        key = random.choice(list(transformations)) #randomly choosing method to call
        img, label = transformations[key](img, label)
        n = n + 1
    return img, label

def make_subfolder(dirname,parent_path):
    path = os.path.join(parent_path, dirname)
    os.mkdir(path)
    print("Directory '%s' created" %dirname)
    return path + '/'

def get_eyeshape(img):
    '''
    Gets rgb image. Return eyeshape mask
    '''
    gray=util.rgb2gray(img)
    binary=gray>15
    return binary

def crop_and_resize_images(img_in_dir, img_out_dir, w, h):
    entries = sorted(os.listdir(img_in_dir))
    for entry in entries:
        img_in = io.imread(img_in_dir + entry)
        img_cropped = img_in[: , 200 : img_in.shape[1] - 550]
        square_img = np.zeros([3538, 3538, 3], dtype = np.uint8)
        square_img[345:3193,:,:] = img_cropped[:,:,0:3]
        img_resized = img_as_ubyte(resize(square_img,(w,h)))
        entry_name = entry.replace('tif', 'png')
        io.imsave(img_out_dir + entry_name.replace('jpg', 'png'), img_resized, check_contrast=False)
        
def create_dataset(idrid_path, datasets_directory, bens, multi_channel, w, h):
    parent_dir = datasets_directory
    bens_name = ''
    if(bens == True):
        bens_name = '_bens'
    ch_name = ''
    if(multi_channel == True):
        ch_name = '_4ch'
    new_directory = 'dataset_' + str(w) + 'x' + str(h) + bens_name + ch_name
    dataset_path = os.path.join(parent_dir, new_directory)
    os.mkdir(dataset_path) 
    print("Directory '%s' created" %new_directory)
    with open(dataset_path + "/info.txt", "w") as file:
        file.write("This dataset is created from the following path: \n")
        file.write(idrid_path)
        file.write("\nWidth of new dataset images: ")
        file.write(str(w))
        file.write("\nHeight of new dataset images: ")
        file.write(str(h))
        file.write("\nInclude bens_processing outputs: ")
        file.write(str(bens))
        file.write("\nInclude 4 channel outputs: ")
        file.write(str(multi_channel))
        file.write("\n\nNote: In this version, it crops the first 200 columns and the last 550 columns from the original images.")
        file.write("\n And adds 390 rows both to the top and to the bottom. Eventually it obtains square shaped images")
    
    org_img = '1. Original Images'
    org_img_path = make_subfolder(org_img,dataset_path)
    
    train_dir = 'a. Training Set'
    org_img_train_path = make_subfolder(train_dir,org_img_path)
    crop_and_resize_images(idrid_path + '/' + org_img + '/' + train_dir + '/', org_img_train_path, w, h)
    
    test_dir = 'b. Testing Set'
    org_img_test_path = make_subfolder(test_dir,org_img_path)
    crop_and_resize_images(idrid_path + '/' + org_img + '/' + test_dir + '/', org_img_test_path, w, h)
    
    if(bens == True):
        bens_train_dir = 'c. Bens Training Set'
        org_img_bens_train_path = make_subfolder(bens_train_dir,org_img_path)
        crop_and_resize_images(idrid_path + '/' + org_img + '/' + bens_train_dir + '/', org_img_bens_train_path, w, h)
        
        bens_test_dir = 'd. Bens Testing Set'
        org_img_bens_test_path = make_subfolder(bens_test_dir,org_img_path)
        crop_and_resize_images(idrid_path + '/' + org_img + '/' + bens_test_dir + '/', org_img_bens_test_path, w, h)
    
    grounds = '2. All Segmentation Groundtruths'
    grounds_path = make_subfolder(grounds,dataset_path)
    
    grounds_train_path = make_subfolder(train_dir,grounds_path)
    
    ma = '1. Microaneurysms'
    grounds_train_ma_path = make_subfolder(ma,grounds_train_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + train_dir + '/' + ma + '/', grounds_train_ma_path, w, h)
    
    he = '2. Haemorrhages'
    grounds_train_he_path = make_subfolder(he,grounds_train_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + train_dir + '/' + he + '/', grounds_train_he_path, w, h)
    
    ex = '3. Hard Exudates'
    grounds_train_ex_path = make_subfolder(ex,grounds_train_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + train_dir + '/' + ex + '/', grounds_train_ex_path, w, h)
    
    se = '4. Soft Exudates'
    grounds_train_se_path = make_subfolder(se,grounds_train_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + train_dir + '/' +se + '/', grounds_train_se_path, w, h)
    
    od = '5. Optic Disc'
    grounds_train_od_path = make_subfolder(od,grounds_train_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + train_dir + '/' + od + '/', grounds_train_od_path, w, h)
    
    if(multi_channel == True):
        ch = '0. All'
        grounds_train_ch_path = make_subfolder(ch,grounds_train_path)
        util.create_4ch_dataset(grounds_train_ma_path, grounds_train_he_path, grounds_train_ex_path, grounds_train_ch_path)
    
    
    grounds_test_path = make_subfolder(test_dir,grounds_path)
    
    ma = '1. Microaneurysms'
    grounds_test_ma_path = make_subfolder(ma,grounds_test_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + test_dir + '/' + ma + '/', grounds_test_ma_path, w, h)
    
    he = '2. Haemorrhages'
    grounds_test_he_path = make_subfolder(he,grounds_test_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + test_dir + '/' + he + '/', grounds_test_he_path, w, h)
    
    ex = '3. Hard Exudates'
    grounds_test_ex_path = make_subfolder(ex,grounds_test_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + test_dir + '/' + ex + '/', grounds_test_ex_path, w, h)
    
    se = '4. Soft Exudates'
    grounds_test_se_path = make_subfolder(se,grounds_test_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + test_dir + '/' +se + '/', grounds_test_se_path, w, h)
    
    od = '5. Optic Disc'
    grounds_test_od_path = make_subfolder(od,grounds_test_path)
    crop_and_resize_images(idrid_path + '/' + grounds + '/' + test_dir + '/' + od + '/', grounds_test_od_path, w, h)
    
    if(multi_channel == True):
        ch = '0. All'
        grounds_test_ch_path = make_subfolder(ch,grounds_test_path)
        util.create_4ch_dataset(grounds_test_ma_path, grounds_test_he_path, grounds_test_ex_path, grounds_test_ch_path, ind = 55)

def patch_image(img, img_name, window_size, stride, output_patch_path):
    name, img_format = img_name.split(".")
    try:
        idridd, indexx, typee = name.split("_")
    except:
        idridd, indexx = name.split("_")
        typee = ""
    
    print(name)
    for i in range(ceil(img.shape[0]/stride)):
        for j in range(ceil(img.shape[1]/stride)):
            x_start, y_start = i*stride, j*stride
            if(img.ndim==2):
                img = img.reshape((img.shape[0], img.shape[1], 1))
            patch_in = img[x_start:x_start+window_size, y_start:y_start+window_size,:]
            draft = np.zeros([window_size, window_size, 3], dtype = np.uint8)
            index_x = str(i).zfill(3)
            index_y = str(j).zfill(3)
            draft[0:patch_in.shape[0], 0:patch_in.shape[1]] = img_as_ubyte(patch_in[:,:,0:3])
            io.imsave(output_patch_path + idridd + "_" + indexx + "_" + index_x + "_" + index_y + "_" + typee + '.png', draft, check_contrast=False)

def patch_folder(img_in_dir, img_out_dir, window_size, stride):
    entries = sorted(os.listdir(img_in_dir))
    for entry in entries:
        img=io.imread(img_in_dir+entry)
        patch_image(img, entry, window_size, stride, img_out_dir)

def patch_dataset(original_dataset_path,datasets_directory, window_size, stride):
    parent_dir = datasets_directory
    new_directory = 'Patched_dataset_' + str(window_size) + 'x' + str(window_size) + '_stride_' + str(stride)
    dataset_path = os.path.join(parent_dir, new_directory)
    '''
    os.mkdir(dataset_path) 
    print("Directory '%s' created" %new_directory)
    with open(dataset_path + "/info.txt", "w") as file:
        file.write("This dataset is created from the following path: \n")
        file.write(original_dataset_path)
        file.write("\nPatch window size of new dataset images: ")
        file.write(str(window_size))
        file.write("\nPatch stride of new dataset images: ")
        file.write(str(stride))

    org_img = '1. Original Images'
    org_img_path = make_subfolder(org_img,dataset_path)
    
    train_dir = 'a. Training Set'
    org_img_train_path = make_subfolder(train_dir,org_img_path)
    patch_folder(original_dataset_path + '/' + org_img + '/' + train_dir + '/', org_img_train_path, window_size, stride)
    
    test_dir = 'b. Testing Set'
    org_img_test_path = make_subfolder(test_dir,org_img_path)
    patch_folder(original_dataset_path + '/' + org_img + '/' + test_dir + '/', org_img_test_path, window_size, stride)
    
    bens_train_dir = 'c. Bens Training Set'
    org_img_bens_train_path = make_subfolder(bens_train_dir,org_img_path)
    patch_folder(original_dataset_path + '/' + org_img + '/' + bens_train_dir + '/', org_img_bens_train_path, window_size, stride)
    
    bens_test_dir = 'd. Bens Testing Set'
    org_img_bens_test_path = make_subfolder(bens_test_dir,org_img_path)
    patch_folder(original_dataset_path + '/' + org_img + '/' + bens_test_dir + '/', org_img_bens_test_path, window_size, stride)
    
    grounds = '2. All Segmentation Groundtruths'
    grounds_path = make_subfolder(grounds,dataset_path)
    
    grounds_train_path = make_subfolder(train_dir,grounds_path)
    
    ma = '1. Microaneurysms'
    grounds_train_ma_path = make_subfolder(ma,grounds_train_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + train_dir + '/' + ma + '/', grounds_train_ma_path, window_size, stride)
    
    he = '2. Haemorrhages'
    grounds_train_he_path = make_subfolder(he,grounds_train_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + train_dir + '/' + he + '/', grounds_train_he_path, window_size, stride)
    
    ex = '3. Hard Exudates'
    grounds_train_ex_path = make_subfolder(ex,grounds_train_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + train_dir + '/' + ex + '/', grounds_train_ex_path, window_size, stride)
    
    se = '4. Soft Exudates'
    grounds_train_se_path = make_subfolder(se,grounds_train_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + train_dir + '/' + se + '/', grounds_train_se_path, window_size, stride)
    
    od = '5. Optic Disc'
    grounds_train_od_path = make_subfolder(od,grounds_train_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + train_dir + '/' + od + '/', grounds_train_od_path, window_size, stride)

    #bv = '6. Blood Vessel'
    #grounds_train_bv_path = make_subfolder(bv,grounds_train_path)
    #patch_folder(original_dataset_path + '/' + grounds + '/' + train_dir + '/' + bv + '/', grounds_train_bv_path, window_size, stride)
    
    ch = '0. All'
    grounds_train_ch_path = make_subfolder(ch,grounds_train_path)
    '''
    grounds_train_path = '/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/'
    grounds_train_ma_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/'
    grounds_train_he_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages/'
    grounds_train_ex_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/'
    grounds_train_ch_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/0. All/'

    rows = util.patch_4ch_dataset(grounds_train_ma_path, grounds_train_he_path, grounds_train_ex_path, grounds_train_ch_path)
    #rows = util.patch_6ch_dataset(grounds_train_ma_path, grounds_train_he_path, grounds_train_ex_path, grounds_train_bv_path, grounds_train_ch_path)
    filename = "/train.csv"
    #fields = ['Input', 'MA', 'HE', 'EX', 'BV', '4CH']
    fields = ['Input', 'MA', 'HE', 'EX', '4CH', 'Background']
    with open(dataset_path + filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows) 
    
    '''
    grounds_test_path = make_subfolder(test_dir,grounds_path)
    
    ma = '1. Microaneurysms'
    grounds_test_ma_path = make_subfolder(ma,grounds_test_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + test_dir + '/' + ma + '/', grounds_test_ma_path, window_size, stride)
    
    he = '2. Haemorrhages'
    grounds_test_he_path = make_subfolder(he,grounds_test_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + test_dir + '/' + he + '/', grounds_test_he_path, window_size, stride)
    
    ex = '3. Hard Exudates'
    grounds_test_ex_path = make_subfolder(ex,grounds_test_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + test_dir + '/' + ex + '/', grounds_test_ex_path, window_size, stride)
    
    se = '4. Soft Exudates'
    grounds_test_se_path = make_subfolder(se,grounds_test_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + test_dir + '/' +se + '/', grounds_test_se_path, window_size, stride)
    
    od = '5. Optic Disc'
    grounds_test_od_path = make_subfolder(od,grounds_test_path)
    patch_folder(original_dataset_path + '/' + grounds + '/' + test_dir + '/' + od + '/', grounds_test_od_path, window_size, stride)

    #bv = '6. Blood Vessel'
    #grounds_test_bv_path = make_subfolder(bv,grounds_test_path)
    #patch_folder(original_dataset_path + '/' + grounds + '/' + test_dir + '/' + bv + '/', grounds_test_bv_path, window_size, stride)
    
    ch = '0. All'
    grounds_test_ch_path = make_subfolder(ch,grounds_test_path)
    '''
    grounds_test_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/b. Testing Set/'
    grounds_test_ma_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms/'
    grounds_test_he_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/b. Testing Set/2. Haemorrhages/'
    grounds_test_ex_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates/'
    grounds_test_ch_path='/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/b. Testing Set/0. All/'
    rows = util.patch_4ch_dataset(grounds_test_ma_path, grounds_test_he_path, grounds_test_ex_path, grounds_test_ch_path)
    #rows = util.patch_6ch_dataset(grounds_test_ma_path, grounds_test_he_path, grounds_test_ex_path, grounds_test_bv_path, grounds_test_ch_path)
    filename = "/test.csv"
    #fields = ['Input', 'MA', 'HE', 'EX', 'BV', '4CH']
    fields = ['Input', 'MA', 'HE', 'EX', '4CH', 'Background']
    with open(dataset_path + filename, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows) 

'''
idrid_path = r'/home/vivente/Downloads/A.Segmentation_adems/'
new_dataset_dir = r'/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/'
datafolder_2048 = r'/home/vivente/Desktop/TEZ/Dataset/A. Segmentation/'

#create_dataset(idrid_path, new_dataset_dir, bens = True, multi_channel = True, w=2048, h=2048)
patch_dataset(datafolder_2048,new_dataset_dir, window_size = 512, stride = 256)
#patch_dataset(esma,new_dataset_dir, window_size = 256, stride = 128)
'''
