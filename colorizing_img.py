import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
import skimage.io as io
import numpy as np
import tensorflow as tf
from skimage.io import imshow
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
import matplotlib.pyplot as plt

def calculate_image(image, model):

    model = load_model(model)    

    img1_color=[]
    img1=img_to_array(load_img(image))
    img1 = resize(img1 ,(256,256))
    img1_color.append(img1)
    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
    test = img1_color
    img1_color = img1_color.reshape(img1_color.shape+(1,))
    output1 = model.predict([img1_color, img1_color])
    output1 = output1*128
    result = np.zeros((256, 256, 3))
    result[:,:,0] = img1_color[0][:,:,0]
    result[:,:,1:] = output1[0]


    return lab2rgb(result)
     

