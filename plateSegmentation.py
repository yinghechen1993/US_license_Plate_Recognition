'''
CSE 415 Final Project
Yinghe Chen, Yu Fu
June 2, 2015
plateSegmentation.py

This program will scan an folder, and try to recognize and segment the characters
on the license plate to individual images. One square image for each character.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy import ndimage
from scipy import misc

from skimage import data
from skimage import io
from skimage import novice
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import os
import sys

def pad_to_square(a, pad_value=1):
    '''Pad a image to square image. Fill in with given padding value'''
    h = a.shape[0]
    w = a.shape[1]
    m = a.reshape((h, -1))
    padded = pad_value * np.ones(2 * [h], dtype=m.dtype)
    padded[0:h, (h-w)/2:(h-w)/2+w] = m
    return padded

def img_seg(img_path):
    '''Try to recognize and segment the characters on given license plate image.'''
    img = io.imread(img_path, 1)

    # Apply Thresholding using Otsu method
    val = threshold_otsu(img)
    bw = closing(img < val, square(3))

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    image_label_overlay = label2rgb(label_image, image=img)

    # Calculate height range of character
    heights = []
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        heights.append(height)

    list.sort(heights)
    list.reverse(heights)

    max_height = heights[0]
    min_height = max_height + 1
    eps = img.shape[1] * 0.01
    count = 0
    for i in range(len(heights) - 1):
        min_height = heights[i+1]
        if max_height-min_height >= eps:
            count = 0
            max_height = min_height
        else:
            count = count + 1
            if count == 5:
                break
    max_height = max_height + eps
    min_height = min_height - eps

    # Calculate width range of character
    widths = []
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        if height <= max_height and height >= min_height:
            widths.append(width)
    list.sort(widths)
    list.reverse(widths)
    max_width = widths[0]
    count = 1
    while widths[count] * 1.5 < max_width:
        max_width = widths[count]
        count = count + 1

    # Box out characters
    crop_imgs = {}
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc

        if height > max_height or height < min_height or width > max_width:
            continue
        crop_imgs[minc] = bw[minr:maxr, minc:maxc]

    # Output to folder
    dir = img_path.split('.')[0]
    try:
        os.stat(dir)
    except:
        os.mkdir(dir) 

    key_list = list(crop_imgs.keys())
    list.sort(key_list)
    for i in range(len(key_list)):
        misc.imsave(dir + "/" + str(i) + ".jpg", pad_to_square(1-crop_imgs[key_list[i]]))
    print(1-crop_imgs[key_list[i]])

def scan_folder(dir):
    '''Scan and process all the image in the given folder.'''
    for filename in os.listdir(dir):
        print(dir + '/' + filename)
        try:
            img_seg(dir + '/' + filename)
        except:
            pass

scan_folder('Images/seg_test')