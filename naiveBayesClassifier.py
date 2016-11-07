'''
CSE 415 Final Project
Yinghe Chen, Yu Fu
June 2, 2015
naiveBayesClassifier.py

The classifier will randomly read a subset from the training set,
and use naive Bayes method to classify given input images.
The classifier will read all the images from a folder,
recognize the character of the image, and generate an output string
combined with all the characters.
'''

import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import os
import sys
import math
import random

def img_to_array(imgpath):
    '''Return an list of length 900 of given image.
       1 if the block has some black. 0 if the block is all white.'''
    img = misc.imread(imgpath, 1)
    small_img = (misc.imresize(img, (30, 30), interp='bicubic') / 255).astype('int32')
    return (1-small_img.flatten()).tolist()

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

def get_prior_char(training_set):
    '''Calculate and return the priors of characters as a list.'''
    patterns = "0123456789abcdefghijklmnpqrstuvwxyz"
    count = [0] * len(patterns)
    for training_list in training_set:
        count[patterns.index(training_list[1])] += 1
    total_count = sum(count)
    for i in range(len(count)):
        count[i] /= total_count
    return count

def get_prior_attr(training_set):
    '''Calcuate and return the priors of attributes values equal to 1.'''
    count = [0] * 900
    total = len(training_set)
    for training_list in training_set:
        for i in range(900):
            count[i] += training_list[0][i]
    # avoid probability of 0
    for i in range(900):
        if count[i] == 0:
            count[i] += 0.01
            total += 0.01
    return [c / total for c in count]
       
def get_indi(training_set):
    '''Calculate and return the probabilty of individual features.
       [[P(block1|'a')...P(block900|'a')], 'a'], ....]'''
    patterns = "0123456789abcdefghijklmnpqrstuvwxyz"
    indi_prob = []
    for c in patterns:
        count_char = 0
        count_attr_list = [0] * 900
        for training_list in training_set:
            if training_list[1] == c:
                count_char += 1
                for i in range(900):
                    count_attr_list[i] += training_list[0][i]

        for i in range(900):
            if count_attr_list[i] == 0:
                count_attr_list[i] += 0.01
                count_char += 0.01

        indi_prob.append([[count / count_char for count in count_attr_list], c])

    return indi_prob

def get_likelihood(input_list, c, indi_prob, prior_char):
    '''Calculate and return P(c|input)'''
    patterns = "0123456789abcdefghijklmnpqrstuvwxyz"
    char_index = patterns.index(c)
    log_p_input_c = 0
    prob_list = indi_prob[char_index][0]

    for i in range(900):
        if input_list[i] == 0:
            log_p_input_c += math.log(1 - prob_list[i])
        else:
            log_p_input_c += math.log(prob_list[i])
    p_c = prior_char[char_index]
    return log_p_input_c + math.log(p_c)

def get_result(imgpath, indi_prob, prior_char):
    '''Use Naive Bayes classifier to compute 
       the character with max likihood of given image array.'''
    input = img_to_array(imgpath)
    max_log_p = -sys.maxsize - 1
    max_c = ''
    for c in "0123456789abcdefghijklmnpqrstuvwxyz":
        log_p = get_likelihood(input, c, indi_prob, prior_char)
        if log_p > max_log_p:
            max_log_p = log_p
            max_c = c
    return (max_c, max_log_p)

def print_array(l):
    '''test function to properly print the array'''
    list = l[:]
    for i in range(900):
        if list[i] >= 0.3:
            list[i] = "*"
        else:
            list[i] = " "
        
    for i in range(30):
        for j in range(30):
            print(list[30 * i + j], end = ' ')
        print('')

def imgToString(img_dir):
    '''Read images from a directory, process all the images and return the result as a string.'''
    training_set = load_training_set('Images/training_sets') # Change the training set here
    prior_char = get_prior_char(training_set)
    prior_attr = get_prior_attr(training_set)
    indi_prob = get_indi(training_set)
    result = ''
    for filename in os.listdir(img_dir):
        #print(get_result(img_dir + '/' + filename, indi_prob, prior_char))
        result += get_result(img_dir + '/' + filename, indi_prob, prior_char)[0]
    return result