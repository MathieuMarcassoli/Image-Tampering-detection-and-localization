import os

import cv2
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize as skresize
from tqdm import tqdm

q = [4.0, 12.0, 2.0]
filter1 = [[0, 0, 0, 0, 0],
           [0, -1, 2, -1, 0],
           [0, 2, -4, 2, 0],
           [0, -1, 2, -1, 0],
           [0, 0, 0, 0, 0]]
filter2 = [[-1, 2, -2, 2, -1],
           [2, -6, 8, -6, 2],
           [-2, 8, -12, 8, -2],
           [2, -6, 8, -6, 2],
           [-1, 2, -2, 2, -1]]
filter3 = [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]

filter1 = np.asarray(filter1, dtype=float) / q[0]
filter2 = np.asarray(filter2, dtype=float) / q[1]
filter3 = np.asarray(filter3, dtype=float) / q[2]

filters = filter1 + filter2 + filter3

def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode not in ('RGB'):  # or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGB", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        im = bg
    return im

def generate_dataset(ds_path):
    data_list = os.listdir(ds_path)
    tampered_image = []
    processed_image = []
    mask_image = []
    for dataset in data_list:
        fakes= os.listdir(ds_path + str(dataset)) 
        for x in tqdm(fakes):
            try:
                if '.mask.' in x or '_mask.' in x or '_gt.' in x:
                    continue
                image = Image.open(ds_path + str(dataset)+'/' + x)
                image = remove_transparency(image)
                img = np.asarray(image)  # imread(ds_path + x)
                
                img = skresize(img, (512, 512, 3))
                tampered_image.append(img)
            except Exception as ex:
                print( x , ex)
            
                #img = imread(ds_path + str(dataset)+'/' + x)
                #img = skresize(img, (512, 512, 3))
                #tampered_image.append(img)
    
        
        for x in tqdm(fakes):
            if '.mask.' in x or '_mask.' in x or '_gt.' in x:
                continue
            image = Image.open(ds_path + str(dataset)+'/' + x)
            image = remove_transparency(image)
            img = np.asarray(image)# imread(ds_path + x)
            img = skresize(img, (512, 512, 3))
            processed_image.append(cv2.filter2D(img, -1, filters))
            
            # img = imread(ds_path + str(dataset)+'/' + x)
            # img = skresize(img, (512, 512, 3))
            # processed_image.append(cv2.filter2D(img, -1, filters))
    
        for x in tqdm(fakes):
            try:
                if '.mask.' in x or '_mask.' in x or '_gt.' in x:
                    image = Image.open(ds_path + str(dataset)+'/' + x)
                    image = remove_transparency(image)
                    img = np.asarray(image)  # imread(ds_path + x)
    
                    img = skresize(img, (512, 512, 1))
                    mask_image.append(img)
            except Exception as ex:
                print( x , ex)
                
                # img = imread(ds_path + str(dataset)+'/' + x)
                # img = skresize(img, (512, 512, 1))
                # mask_image.append(img)

    X1 = np.array(tampered_image)
    X2 = np.array(processed_image)
    Y = np.array(mask_image)
    return X1, X2, Y

