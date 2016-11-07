#Final Project
#Yinghe Chen, Yu Fu
#perceptronClassifier.py
#CSE 415
#June2, 2015
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import os
import sys
import random

# load image as 1d array of 1 or 0
def img_to_array(imgpath):
    img = misc.imread(imgpath, 1)
    small_img = (misc.imresize(img, (20, 20), interp = 'bicubic') /255).astype('int32')
    return small_img.flatten()
# load the folder which contains training set
def load_training_set(path):
    '''Load and return training set from given path.'''
    training_set = []
    patterns = "0123456789abcdefghijklmnpqrstuvwxyz"
    for c in patterns:
        pattern_dir = path + '/' + c
        rand = 1.0
        for filename in os.listdir(pattern_dir):
            if rand >= 0.2:  # Change this number to modify the size of subset
                training_set.append([img_to_array(pattern_dir + '/' + filename), c])
            rand = random.random()
    return training_set
# conduct perceptron classifer process and return single match result as letter
def perceptron(img, path):
    list = []
    picToExam = img_to_array(img)
    traning_set = load_training_set(path)
    max = 0
    letter = "a"
    for i in traning_set:
        list.append(perceptron_process(picToExam, i))
    for j in list:
        if (j[0]> max):
            letter = j[1]
            max = j[0]
    return letter
# conduct perceptron training
def perceptron_process(img_array1, training_element):
    w = []
    for i in range (0, len(img_array1) - 1):
        if (i % 20 < 15 and i % 20 >20):
            w.append(1.5)
        else:
            w.append(1)
    beta = 0
    a = 0
    for i in range (0, len(img_array1) - 1):
        theta = abs(training_element[0][i] - img_array1[i])
        if (theta < 1/2):
            beta += w[i]
    return [beta, training_element[1]]
# process a set of pictures and return the recognized license plate
def imgToString(img_dir):
    result = ""
    for filename in os.listdir(img_dir):
        filename = img_dir + '/' + filename
        result += perceptron(filename, 'Images/training_sets')
    return result

#print(imgToString('Images/seg_test/123'))