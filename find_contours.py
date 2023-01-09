import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import time

def read_var(file_name):   
    infile = open(file_name,'rb')
    var = pickle.load(infile)
    infile.close()
    return var
def draw_rectangle(arr, x1, y1, width, height):
    arr[x1:x1+width,y1:y1+height] = 255
    arr = arr.astype(np.uint8)
    return arr
def create_random_arr():
    arr= np.random.randint(2, size=(512, 512), dtype = np.uint8)

def create_rectangle_box_array():
    arr = np.zeros((512,512), dtype=np.uint8)
    arr = draw_rectangle(arr, 50,100,100,20)
    arr = draw_rectangle(arr, 300,300,50,100)

def isOverlapping1D(xmin1, xmax1, xmin2, xmax2):
    return xmax1 >= xmin2 and xmax2 >= xmin1

def isOverlapping2D(xmin1,xmax1,ymin1,ymax1, xmin2,xmax2,ymin2,ymax2):
    return isOverlapping1D(xmin1,xmax1, xmin2,xmax2) and isOverlapping1D(ymin1,ymax1, ymin2,ymax2)

file_name = '/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/0. All/IDRiD_01_004_009_4CH'
tmp = np.asarray(read_var(file_name)).astype(np.uint8)*255
arr_pred= np.array(tmp[:,:,1].reshape((512,512)))

file_name = '/home/vivente/Desktop/TEZ/Dataset/Patch Lesion/Patched_dataset_512x512_stride_256/2. All Segmentation Groundtruths/a. Training Set/0. All/IDRiD_01_004_010_4CH'
tmp = np.asarray(read_var(file_name)).astype(np.uint8)*255
arr_true= np.array(tmp[:,:,1].reshape((512,512)))

tmp_2 = np.asarray(read_var(file_name)).astype(np.uint8)*255
img = np.array(tmp_2[:,:,:1].reshape((512,512)))

#im2, contours_pred, hierarchy = cv2.findContours(arr_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#_, contours_true, _ = cv2.findContours(arr_true, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

plt.imshow(arr_pred, cmap='gray')
plt.show()

plt.imshow(arr_true, cmap='gray')
plt.show()

#print("Number of Contours Groundtruth found = " + str(len(contours_true)))
#print("Number of Contours Prediction found = " + str(len(contours_pred)))


intersection = arr_pred * arr_true
union = arr_pred + arr_true - intersection
#Find contours of every intance in the UNION SET 
output = cv2.connectedComponentsWithStats(union, 1, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

plt.imshow(union, cmap='gray')
plt.show()
# initialize an output mask to store all characters parsed from
# the license plate
mask = np.zeros((512, 512), dtype="uint8")
# loop over the number of unique connected component labels, skipping
# over the first label (as label zero is the background)
for i in range(1, numLabels):
    # construct a mask for the current connected component and
    # then take the bitwise OR with the mask
    print("[INFO] keeping connected component '{}'".format(i))
    componentMask = (labels == i).astype("uint8") * 255
    plt.imshow(componentMask)
    plt.show()
    mask = cv2.bitwise_or(mask, componentMask)
# show the original input image and the mask for the license plate
# characters
plt.imshow(mask)
plt.show()
cv2.imshow("Characters", mask)
cv2.waitKey(0)

'''
# Draw all contours
# -1 signifies drawing all contours
contours_inst = contours[0]
for cont in contours_inst:
    print(cont[0])
    t=cont[0]
    img[t[1], t[0]] = 127
'''
#cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
'''
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(arr)
axs[1].imshow(img)  
#plt.imshow(img) #Contours
plt.show()
time.sleep(5)
'''