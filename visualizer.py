import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage as sk
from skimage.transform import resize
import pandas as pd
import random
import time

from tensorflow.keras.utils import to_categorical   
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import augmentation as augment
import utils as util
from utils import rgb2gray, rgb2bool, feat_train_config, make_subfolder, make_log_dir, write_to_log, save_var, save_model
from model import residual_unet, dice_loss, dice_coef_acc
from feat_train_functions import *

import glob
from PIL import Image

def make_gif(frame_folders):
    frames = []
    entries = sorted(os.listdir(frame_folders[0]))
    for entry in entries:
        frames = []
        for folder in frame_folders:
            img_in = Image.open(folder + entry)
            frames.append(img_in)
        frame_one = frames[0]
        frame_one.save("out/Gifs/"+entry.replace('png', 'gif'), format="GIF", append_images=frames,
               save_all=True, duration=200, loop=0)
    
if __name__ == "__main__":
    imgs_dirs = []
    imgs_dirs.append("out/2022_January_17-17_12_41/visualisations/")
    imgs_dirs.append("out/2022_January_17-17_16_08/visualisations/")
    imgs_dirs.append("out/2022_January_17-18_10_49/visualisations/")
    imgs_dirs.append("out/2022_January_17-18_38_34/visualisations/")

    make_gif(imgs_dirs)