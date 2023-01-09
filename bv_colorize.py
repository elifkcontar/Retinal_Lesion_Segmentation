#import skimage.io as io
import numpy as np 
#from PIL import Image
import cv2 

def gray2bool(gray):
    bool_img = gray > 0
    return bool_img

def rgb2bool(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    bool_img = (r + g + b) > 0
    return bool_img

bv = rgb2bool(cv2.imread('/Users/elifkcontar/Desktop/IDRiD_55_BV.png'))
ma = rgb2bool(cv2.imread('/Users/elifkcontar/Desktop/IDRiD_55_MA.tif'))
he = rgb2bool(cv2.imread('/Users/elifkcontar/Desktop/IDRiD_55_HE.tif'))
ex = rgb2bool(cv2.imread('/Users/elifkcontar/Desktop/IDRiD_55_EX.tif'))

lesions = ma | he | ex
orig_img = cv2.imread('/Users/elifkcontar/Desktop/IDRiD_55_orig.jpg').astype('float64')
h, w = orig_img.shape[:2]
# Add an alpha channel, fully opaque (255)
#RGBA = np.dstack((orig_img, np.zeros((h,w),dtype=np.uint8)+255))

fp = bv & lesions
overlay = np.zeros((orig_img.shape))
overlay[fp] = [255,0,0]
bv_other = bv & (~fp)
overlay2 = np.zeros((orig_img.shape))
overlay2[bv_other] = [0,255,0]


cv2.addWeighted(overlay, 0.7, orig_img, 1,
		0, orig_img)
cv2.addWeighted(overlay2, 0.4, orig_img, 1,
		0, orig_img)
#RGBA[fp] = [0, 0, 255, 10]
#RGBA[bv_other] = [0, 255, 0, 10]
#Image.fromarray(RGBA).save('result.png')
cv2.imwrite('result.png', orig_img)


